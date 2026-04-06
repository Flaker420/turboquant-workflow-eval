from __future__ import annotations

import gc
import math
import random
from pathlib import Path
from typing import Any

import torch

from .code_runner import extract_python_code, run_code_with_tests
from .generation import generate_one
from .import_utils import load_object
from .loader import load_model_module, load_policy_module, load_study_module
from .model_loader import load_model_and_tokenizer
from .prompts import filter_prompts, load_prompt_pack
from .reporting import (
    IncrementalWriter,
    write_csv,
    write_examples_markdown,
    write_jsonl,
    write_run_summary,
    write_text_outputs,
)
from .schema import (
    EarlyStopConfig,
    PolicyConfig,
    StudyConfig,
    ThresholdsConfig,
    model_to_legacy_dict,
    policy_to_legacy_dict,
    replace_path,
)
from .scoring import check_reference_answer, compute_semantic_similarity, compute_verdict
from .types import StudyContext
from .events import EventBus, StudyEvent


# ---------------------------------------------------------------------------
# Study controller (pause / resume / skip / stop / early-stop)
# ---------------------------------------------------------------------------

class StudyController:
    """Controls study execution flow — pause, resume, skip, stop, early-stop."""

    def __init__(
        self,
        event_bus: EventBus | None = None,
        early_stop_cfg: EarlyStopConfig | None = None,
    ) -> None:
        self.paused = False
        self.skip_current_policy = False
        self.stop_requested = False
        self._event_bus = event_bus
        self._early_stop = early_stop_cfg or EarlyStopConfig()
        self._red_count = 0
        self._error_count = 0
        self._prompt_count = 0

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def skip_policy(self) -> None:
        self.skip_current_policy = True

    def stop(self) -> None:
        self.stop_requested = True

    def check_early_stop(self, row: dict) -> bool:
        """Check early-stop conditions after a prompt completes.

        Returns True if the study should stop.
        """
        self._prompt_count += 1
        if row.get("error"):
            self._error_count += 1
        if row.get("verdict") == "red":
            self._red_count += 1

        max_red = self._early_stop.max_red_verdicts
        if max_red is not None and self._red_count >= max_red:
            if self._event_bus:
                self._event_bus.emit_new("early_stop", reason=f"Hit {self._red_count} red verdicts (max: {max_red})")
            return True

        max_error_rate = self._early_stop.max_error_rate
        if max_error_rate is not None and self._prompt_count > 0:
            if self._error_count / self._prompt_count > max_error_rate:
                if self._event_bus:
                    self._event_bus.emit_new("early_stop", reason=f"Error rate {self._error_count}/{self._prompt_count} exceeds {max_error_rate}")
                return True

        return False

    def reset_policy_state(self) -> None:
        """Reset per-policy counters."""
        self.skip_current_policy = False

    def wait_if_paused(self) -> None:
        """Block until resumed. Called between prompts."""
        import time
        while self.paused and not self.stop_requested:
            time.sleep(0.1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _aggregate_stats(values: list[float]) -> dict[str, float]:
    """Compute mean and std for a list of values."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "std": 0.0}
    mean = sum(values) / n
    if n < 2:
        return {"mean": mean, "std": 0.0}
    variance = sum((x - mean) ** 2 for x in values) / (n - 1)
    return {"mean": mean, "std": math.sqrt(variance)}


# ---------------------------------------------------------------------------
# Phase 1: Prepare study context (no GPU)
# ---------------------------------------------------------------------------

def prepare_study(
    study: StudyConfig | str | Path,
    output_dir: str | Path,
    *,
    prompt_ids: list[str] | None = None,
    prompt_categories: list[str] | None = None,
    prompt_pattern: str | None = None,
    single_mode: bool = False,
) -> StudyContext:
    """Resolve a :class:`StudyConfig` into a runnable :class:`StudyContext`.

    Loads the prompt pack(s), applies prompt filters, optionally trims to the
    first enabled policy for ``single_mode``, optionally shuffles the policy
    order, and returns a context object. No GPU is touched.

    Accepts either an already-constructed ``StudyConfig`` (the modern path,
    used by the CLI) or a path to a Python config module (convenience for
    test code and other callers that want one-shot loading).
    """
    if isinstance(study, (str, Path)):
        study = load_study_module(study)

    # Load prompts
    prompt_pack: list = []
    for pp in study.prompt_pack:
        prompt_pack.extend(load_prompt_pack(pp))

    # Filter prompts
    has_filter = prompt_ids is not None or prompt_categories is not None or prompt_pattern is not None
    if has_filter or single_mode:
        prompt_pack = filter_prompts(
            prompt_pack,
            prompt_ids=prompt_ids,
            categories=prompt_categories,
            pattern=prompt_pattern,
        )
    if single_mode and prompt_pack:
        prompt_pack = prompt_pack[:1]
    if not prompt_pack:
        raise RuntimeError("No prompts matched the specified filters.")

    # Optionally shuffle policy order (in-place on the StudyConfig.policies tuple).
    if study.runtime.shuffle_policies:
        rng = random.Random(study.runtime.shuffle_seed)
        ordered = list(study.policies)
        rng.shuffle(ordered)
        study = replace_path(study, "policies", tuple(ordered))

    # In single-mode, trim policies to the first enabled one. This mirrors
    # the legacy behaviour where ``--single`` runs exactly one policy on
    # exactly one prompt.
    if single_mode:
        for p in study.policies:
            if p.enabled:
                study = replace_path(study, "policies", (p,))
                # Single-policy studies must declare themselves as their own
                # baseline so score_results doesn't trip on the multi-policy
                # baseline guard.
                study = replace_path(study, "baseline_policy_name", p.name)
                break

    return StudyContext(
        study=study,
        prompt_pack=prompt_pack,
        output_dir=Path(output_dir),
    )


# ---------------------------------------------------------------------------
# Phase 2: Run a single prompt
# ---------------------------------------------------------------------------

def _runtime_to_dict(runtime) -> dict[str, Any]:
    """Convert RuntimeConfig to a plain dict for ``generate_one``."""
    from dataclasses import asdict
    return asdict(runtime)


def _run_single_prompt(
    model: Any,
    tokenizer: Any,
    prompt: Any,
    policy: PolicyConfig,
    adapter: Any,
    runtime,
    repetitions: int,
) -> dict[str, Any]:
    """Run a single prompt through the model and collect metrics."""
    runtime_cfg = _runtime_to_dict(runtime)

    adapter.reset_generation_state()
    first_result = generate_one(model, tokenizer, prompt.prompt, runtime_cfg, turns=prompt.turns)

    latencies = [first_result["latency_s"]]
    tps_values = [first_result["tokens_per_second"]] if first_result["tokens_per_second"] is not None else []
    vram_values = [first_result["peak_vram_gb"]] if first_result["peak_vram_gb"] is not None else []

    for _ in range(repetitions - 1):
        adapter.reset_generation_state()
        rep_result = generate_one(model, tokenizer, prompt.prompt, runtime_cfg, turns=prompt.turns)
        latencies.append(rep_result["latency_s"])
        if rep_result["tokens_per_second"] is not None:
            tps_values.append(rep_result["tokens_per_second"])
        if rep_result["peak_vram_gb"] is not None:
            vram_values.append(rep_result["peak_vram_gb"])

    lat_stats = _aggregate_stats(latencies)
    tps_stats = _aggregate_stats(tps_values)
    vram_stats = _aggregate_stats(vram_values)

    settings = policy.settings
    row: dict[str, Any] = {
        "policy_name": policy.name,
        "comparison_label": policy.comparison_label,
        "adapter_name": getattr(adapter, "name", adapter.__class__.__name__),
        "prompt_id": prompt.id,
        "category": prompt.category,
        "title": prompt.title,
        "watch_for": prompt.watch_for,
        "prompt_text": prompt.prompt,
        "residual_window": settings.residual_window,
        "key_strategy": settings.key_strategy,
        **first_result,
    }

    if repetitions > 1:
        row["latency_mean"] = lat_stats["mean"]
        row["latency_std"] = lat_stats["std"]
        row["tps_mean"] = tps_stats["mean"]
        row["tps_std"] = tps_stats["std"]
        row["vram_mean"] = vram_stats["mean"]
        row["vram_std"] = vram_stats["std"]
        row["repetitions"] = repetitions

    row["math_correct"] = check_reference_answer(first_result["output_text"], prompt.reference_answer)

    if prompt.test_cases:
        code = extract_python_code(first_result["output_text"])
        if code:
            code_result = run_code_with_tests(code, list(prompt.test_cases))
            row["code_passed"] = code_result["passed"]
            row["code_failed"] = code_result["failed"]
            row["code_errors"] = code_result["errors"]
            row["code_verdict"] = code_result["verdict"]
        else:
            row["code_verdict"] = "error"
            row["code_passed"] = 0
            row["code_failed"] = 0
            row["code_errors"] = 1

    return row


# ---------------------------------------------------------------------------
# Phase 3: Run all prompts for a single policy
# ---------------------------------------------------------------------------

def run_policy(
    ctx: StudyContext,
    policy: PolicyConfig,
    model: Any,
    tokenizer: Any,
    adapter: Any,
    writer: IncrementalWriter | None = None,
    progress_callback: Any | None = None,
    policy_index: int = 0,
    total_policies: int = 1,
    event_bus: EventBus | None = None,
    controller: StudyController | None = None,
    baseline_by_prompt: dict[str, dict] | None = None,
) -> list[dict]:
    """Run all prompts for one policy and return result rows.

    If *baseline_by_prompt* is provided, each row receives a provisional
    verdict (and provisional ``output_length_delta_pct`` /
    ``semantic_similarity``) immediately after generation. This lets the
    early-stop controller and ``prompt_completed`` events fire on a real
    verdict instead of ``None``. ``score_results`` is still called at the
    end of the study to (idempotently) finalize the same fields.
    """
    is_baseline_policy = (
        baseline_by_prompt is not None
        and policy.name == ctx.baseline_policy_name
    )
    thresholds = ctx.thresholds
    rows: list[dict] = []

    if event_bus:
        event_bus.emit_new("policy_started", policy_name=policy.name, policy_index=policy_index)

    for prompt_idx, prompt in enumerate(ctx.prompt_pack):
        # Check controller state
        if controller:
            if controller.stop_requested:
                break
            if controller.skip_current_policy:
                break
            controller.wait_if_paused()

        if progress_callback:
            done = policy_index * len(ctx.prompt_pack) + prompt_idx
            total = total_policies * len(ctx.prompt_pack)
            progress_callback(done / total, f"{policy.name}: {prompt.id}")

        if event_bus:
            event_bus.emit_new("prompt_started", policy_name=policy.name, prompt_id=prompt.id)

        try:
            row = _run_single_prompt(
                model, tokenizer, prompt, policy, adapter, ctx.runtime, ctx.repetitions,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [WARNING] prompt {prompt.id} failed: {exc}")
            settings = policy.settings
            row = {
                "policy_name": policy.name,
                "comparison_label": policy.comparison_label,
                "adapter_name": getattr(adapter, "name", adapter.__class__.__name__),
                "prompt_id": prompt.id,
                "category": prompt.category,
                "title": prompt.title,
                "watch_for": prompt.watch_for,
                "prompt_text": prompt.prompt,
                "residual_window": settings.residual_window,
                "key_strategy": settings.key_strategy,
                "output_text": f"[ERROR] {exc}",
                "output_tokens": 0,
                "latency_s": 0.0,
                "tokens_per_second": None,
                "peak_vram_gb": None,
                "rendered_prompt": "",
                "prompt_tokens": 0,
                "math_correct": None,
                "error": str(exc),
                "verdict": "error",
                "code_verdict": "error",
                "code_passed": 0,
                "code_failed": 0,
                "code_errors": 1,
                "semantic_similarity": None,
                "output_length_delta_pct": 0.0,
            }
            if event_bus:
                event_bus.emit_new("error", policy_name=policy.name, prompt_id=prompt.id, error=str(exc))

        # Provisional scoring so early-stop and prompt_completed events
        # see a real verdict before score_results runs at the end.
        if baseline_by_prompt is not None and not row.get("error"):
            if is_baseline_policy:
                baseline_by_prompt.setdefault(prompt.id, row)
            bl = baseline_by_prompt.get(prompt.id)
            bl_tokens = bl["output_tokens"] if bl else 0
            if bl_tokens and bl_tokens > 0:
                row["output_length_delta_pct"] = (
                    (row["output_tokens"] - bl_tokens) / bl_tokens
                ) * 100
            else:
                row["output_length_delta_pct"] = 0.0
            if bl and bl is not row:
                row["semantic_similarity"] = compute_semantic_similarity(
                    bl["output_text"], row["output_text"]
                )
            else:
                row["semantic_similarity"] = None
            row["verdict"] = compute_verdict(
                row, bl if bl is not row else None, thresholds
            )

        rows.append(row)
        if writer:
            writer.write_row(row)

        if event_bus:
            event_bus.emit_new("prompt_completed", policy_name=policy.name, prompt_id=prompt.id, verdict=row.get("verdict"))

        # Early stopping
        if controller and controller.check_early_stop(row):
            print(f"  [EARLY STOP] Stopping study due to early-stop condition.")
            controller.stop()
            break

    if event_bus:
        event_bus.emit_new("policy_completed", policy_name=policy.name, rows_count=len(rows))

    return rows


# ---------------------------------------------------------------------------
# Phase 4: Post-processing (scoring, verdicts)
# ---------------------------------------------------------------------------

def score_results(
    rows: list[dict],
    thresholds: ThresholdsConfig | None = None,
    baseline_policy_name: str | None = None,
) -> list[dict]:
    """Compute baseline deltas, similarity, and verdicts on existing rows.

    Mutates *rows* in-place and returns them for convenience.

    The baseline row for each prompt is selected by *baseline_policy_name*.
    If only one policy is present in *rows*, that policy is used as the
    baseline by default. Otherwise *baseline_policy_name* is required.
    """
    policy_names = {row.get("policy_name") for row in rows if row.get("policy_name")}

    if baseline_policy_name is None:
        if len(policy_names) == 1:
            baseline_policy_name = next(iter(policy_names))
        else:
            raise ValueError(
                "score_results: multiple policies present "
                f"({sorted(policy_names)!r}) but no baseline_policy_name was specified. "
                "Set 'baseline_policy_name' in the study config."
            )
    elif baseline_policy_name not in policy_names:
        raise ValueError(
            f"score_results: baseline_policy_name={baseline_policy_name!r} "
            f"not found in result rows (present policies: {sorted(policy_names)!r})."
        )

    baseline_by_prompt: dict[str, dict] = {}
    for row in rows:
        if row.get("policy_name") != baseline_policy_name:
            continue
        pid = row["prompt_id"]
        if pid not in baseline_by_prompt:
            baseline_by_prompt[pid] = row

    missing = sorted({row["prompt_id"] for row in rows} - baseline_by_prompt.keys())
    if missing:
        raise ValueError(
            f"score_results: no baseline row for prompt_id(s) {missing!r} "
            f"under baseline policy {baseline_policy_name!r}."
        )

    for pid, bl in baseline_by_prompt.items():
        if bl.get("error") and len(policy_names) > 1:
            raise ValueError(
                f"score_results: baseline row for prompt {pid!r} under policy "
                f"{baseline_policy_name!r} is an error row ({bl['error']!r}); "
                "cannot use as reference."
            )

    for row in rows:
        # Skip rescoring of error rows; their fields were set at creation time.
        if row.get("error"):
            continue

        pid = row["prompt_id"]
        bl = baseline_by_prompt.get(pid)

        bl_tokens = bl["output_tokens"] if bl else 0
        if bl_tokens and bl_tokens > 0:
            row["output_length_delta_pct"] = ((row["output_tokens"] - bl_tokens) / bl_tokens) * 100
        else:
            row["output_length_delta_pct"] = 0.0

        if bl and bl is not row:
            row["semantic_similarity"] = compute_semantic_similarity(bl["output_text"], row["output_text"])
        else:
            row["semantic_similarity"] = None

        row["verdict"] = compute_verdict(row, bl if bl is not row else None, thresholds)

    return rows


# ---------------------------------------------------------------------------
# Phase 5: Write outputs
# ---------------------------------------------------------------------------

def write_results(
    output_dir: Path,
    rows: list[dict],
    study: StudyConfig,
    policies_used: list[dict],
    prompt_count: int,
    repetitions: int,
    study_config_path: Path | None = None,
    baseline_policy_name: str | None = None,
) -> dict:
    """Write all output artefacts and return the summary dict."""
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = study.outputs

    write_jsonl(output_dir / "rows.jsonl", rows)
    write_csv(
        output_dir / "workflow_compare.csv",
        rows,
        truncate_output_to_chars=outputs.truncate_csv_output_to_chars,
    )
    if outputs.write_individual_text_files:
        write_text_outputs(output_dir, rows)
    write_examples_markdown(output_dir / "examples.md", rows)

    verdict_counts = {"green": 0, "yellow": 0, "red": 0}
    for row in rows:
        v = row.get("verdict", "green")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary = {
        "study_name": study.name,
        "study_config": str(study_config_path) if study_config_path else "",
        "model_name": study.model.model_name,
        "policy_count": len(policies_used),
        "prompt_count": prompt_count,
        "row_count": len(rows),
        "repetitions": repetitions,
        "baseline_policy_name": baseline_policy_name,
        "verdict_summary": verdict_counts,
        "policies_used": policies_used,
        "output_dir": str(output_dir),
    }
    write_run_summary(output_dir / "run_summary.json", summary)
    return summary


# ---------------------------------------------------------------------------
# Backward-compatible facade
# ---------------------------------------------------------------------------

def run_workflow_study(
    study: StudyConfig | str | Path,
    output_dir: str | Path,
    *,
    progress_callback: Any | None = None,
    prompt_ids: list[str] | None = None,
    prompt_categories: list[str] | None = None,
    prompt_pattern: str | None = None,
    single_mode: bool = False,
    event_bus: EventBus | None = None,
    controller: StudyController | None = None,
    study_config_path: str | Path | None = None,
) -> dict:
    """Run the full workflow study.

    Orchestrates :func:`prepare_study`, :func:`run_policy`,
    :func:`score_results`, and :func:`write_results`.

    The CLI is responsible for loading the study module, applying any
    ``--set`` / ``--KNOB`` overrides via ``replace_path``, and then passing
    the resolved :class:`StudyConfig` here.

    ``study_config_path`` is purely metadata for the run summary; it does
    not affect loading.
    """
    if isinstance(study, (str, Path)):
        if study_config_path is None:
            study_config_path = Path(study)
        study = load_study_module(study)

    # --- Prepare (no GPU) ---
    ctx = prepare_study(
        study=study,
        output_dir=output_dir,
        prompt_ids=prompt_ids,
        prompt_categories=prompt_categories,
        prompt_pattern=prompt_pattern,
        single_mode=single_mode,
    )

    # --- Event bus and controller ---
    if event_bus is None:
        event_bus = EventBus()
    if controller is None:
        controller = StudyController(event_bus=event_bus, early_stop_cfg=ctx.study.early_stop)

    # Wire progress_callback as an event subscriber
    if progress_callback:
        def _progress_subscriber(event: StudyEvent) -> None:
            pass  # progress_callback is called directly in run_policy
        event_bus.subscribe(_progress_subscriber)

    event_bus.emit_new("study_started", study_name=ctx.study.name)

    # --- Incremental writer ---
    writer = IncrementalWriter(ctx.output_dir)

    rows: list[dict] = []
    policies_used: list[dict] = []
    # Shared baseline lookup for provisional verdicts inside run_policy.
    # Populated as the baseline policy executes; consulted by every policy.
    baseline_by_prompt: dict[str, dict] = {}

    # --- Load model once, reuse across policies ---
    model = None
    tokenizer = None
    loader_name = None
    needs_reload = True

    legacy_model_cfg = model_to_legacy_dict(ctx.study.model)

    for policy_idx, policy in enumerate(ctx.study.policies):
        if not policy.enabled:
            continue

        adapter_cls = load_object(policy.adapter.import_path)
        adapter = adapter_cls()

        # Load or reuse model
        if needs_reload or model is None:
            if model is not None:
                del model
                del tokenizer
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            model, tokenizer, loader_name = load_model_and_tokenizer(legacy_model_cfg)
            needs_reload = False

        legacy_policy_cfg = policy_to_legacy_dict(policy)
        model, tokenizer = adapter.prepare_model(
            model, tokenizer, legacy_model_cfg, legacy_policy_cfg
        )

        policy_metadata = {
            "name": policy.name,
            "comparison_label": policy.comparison_label,
            "adapter_name": getattr(adapter, "name", adapter.__class__.__name__),
            "loader_name": loader_name,
            "adapter_description": adapter.describe(legacy_policy_cfg),
            "import_path": policy.adapter.import_path,
        }
        policies_used.append(policy_metadata)

        # Warmup — prime JIT/CUDA kernels
        generate_one(model, tokenizer, "Say hello.", _runtime_to_dict(ctx.runtime))

        # Run all prompts for this policy
        controller.reset_policy_state()
        policy_rows = run_policy(
            ctx, policy, model, tokenizer, adapter,
            writer=writer,
            progress_callback=progress_callback,
            policy_index=policy_idx,
            total_policies=len(ctx.study.policies),
            event_bus=event_bus,
            controller=controller,
            baseline_by_prompt=baseline_by_prompt,
        )
        rows.extend(policy_rows)

        # Revert or mark for reload
        if adapter.can_revert():
            reverted = adapter.revert(model)
            if not reverted:
                needs_reload = True
        else:
            adapter.cleanup(model)
            needs_reload = True

        if needs_reload and ctx.runtime.cleanup_between_policies:
            pass  # Model will be reloaded at top of next iteration

        if controller.stop_requested:
            break

    writer.close()
    event_bus.emit_new("study_completed", row_count=len(rows))

    if not rows:
        raise RuntimeError("No enabled policies were run. Enable at least one policy config.")

    # --- Post-processing ---
    score_results(rows, ctx.thresholds, baseline_policy_name=ctx.baseline_policy_name)

    # Resolve the effective baseline name for provenance: if the user did not
    # set one and exactly one policy ran, record that policy's name so the
    # summary is never ambiguous about what was scored against what.
    effective_baseline = ctx.baseline_policy_name
    if effective_baseline is None:
        unique_policies = {row.get("policy_name") for row in rows if row.get("policy_name")}
        if len(unique_policies) == 1:
            effective_baseline = next(iter(unique_policies))

    # --- Write final outputs (overwrites incremental JSONL with scored version) ---
    summary = write_results(
        output_dir=ctx.output_dir,
        rows=rows,
        study=ctx.study,
        policies_used=policies_used,
        prompt_count=len(ctx.prompt_pack),
        repetitions=ctx.repetitions,
        study_config_path=Path(study_config_path) if study_config_path else None,
        baseline_policy_name=effective_baseline,
    )

    # Final cleanup
    if model is not None:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary
