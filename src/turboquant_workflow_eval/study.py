from __future__ import annotations

import gc
import math
import random
from pathlib import Path
from typing import Any

import torch

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
    model_to_legacy_dict,
    policy_to_legacy_dict,
    replace_path,
)
from .scoring import compute_divergence, compute_kv_cache_bytes
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

        Returns True if the study should stop. The legacy "red verdict"
        condition (``EarlyStopConfig.max_red_verdicts``) is intentionally
        a no-op in the divergence-metrics world: rows no longer carry a
        verdict, and the meaningful per-row signal is captured by the
        post-hoc divergence + KV-cache scoring rather than mid-run.
        Only the error-rate guard remains live.
        """
        self._prompt_count += 1
        if row.get("error"):
            self._error_count += 1

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
        # Policy knobs persisted on the row so the post-hoc divergence and
        # KV-cache-bytes scoring (run after the full study completes) can
        # reconstruct the per-row policy without having to look it back up
        # against the live PolicyConfig list.
        "bit_width": settings.bit_width,
        "residual_window": settings.residual_window,
        "key_strategy": settings.key_strategy,
        "value_strategy": settings.value_strategy,
        "compressible_layers": (
            list(settings.compressible_layers)
            if settings.compressible_layers is not None else None
        ),
        "compressible_heads": (
            list(settings.compressible_heads)
            if settings.compressible_heads is not None else None
        ),
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

    If *baseline_by_prompt* is provided it is populated with the baseline
    policy's rows as they're produced, so downstream consumers (the early-
    stop controller, the live event stream) can look up baseline outputs
    while later policies are still running. The full divergence + KV-cache
    accounting is computed once by ``score_results`` after every policy
    has finished.
    """
    is_baseline_policy = (
        baseline_by_prompt is not None
        and policy.name == ctx.baseline_policy_name
    )
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
                "bit_width": settings.bit_width,
                "residual_window": settings.residual_window,
                "key_strategy": settings.key_strategy,
                "value_strategy": settings.value_strategy,
                "compressible_layers": (
                    list(settings.compressible_layers)
                    if settings.compressible_layers is not None else None
                ),
                "compressible_heads": (
                    list(settings.compressible_heads)
                    if settings.compressible_heads is not None else None
                ),
                "output_text": f"[ERROR] {exc}",
                "output_token_ids": [],
                "output_tokens": 0,
                "latency_s": 0.0,
                "tokens_per_second": None,
                "peak_vram_gb": None,
                "rendered_prompt": "",
                "prompt_tokens": 0,
                "error": str(exc),
            }
            if event_bus:
                event_bus.emit_new("error", policy_name=policy.name, prompt_id=prompt.id, error=str(exc))

        # Stash baseline rows as soon as they're produced so subsequent
        # policies and live consumers can look them up by prompt_id, even
        # though full divergence/KV-cache scoring runs later.
        if baseline_by_prompt is not None and is_baseline_policy and not row.get("error"):
            baseline_by_prompt.setdefault(prompt.id, row)

        rows.append(row)
        if writer:
            writer.write_row(row)

        if event_bus:
            event_bus.emit_new("prompt_completed", policy_name=policy.name, prompt_id=prompt.id)

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
    baseline_policy_name: str | None = None,
    *,
    model_info: dict[str, int] | None = None,
) -> list[dict]:
    """Annotate every row with divergence-vs-baseline and KV-cache-bytes
    metrics. Mutates *rows* in place; returned for convenience.

    The baseline row for each prompt is selected by *baseline_policy_name*
    (auto-detected when only one policy is present, error otherwise).

    *model_info* must contain ``num_hidden_layers``, ``num_key_value_heads``,
    and ``head_dim``. It is fetched once at model load time in
    ``run_workflow_study`` and threaded through here so that this function
    stays torch-free and unit-testable. When called from ``--rescore`` on a
    cold ``rows.jsonl`` it can be omitted, in which case KV-cache bytes are
    set to ``None`` on every row (divergence is still computed).
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

    can_score_kv = model_info is not None
    if can_score_kv:
        num_layers = int(model_info["num_hidden_layers"])
        num_kv_heads = int(model_info["num_key_value_heads"])
        head_dim = int(model_info["head_dim"])

    for row in rows:
        if row.get("error"):
            continue

        pid = row["prompt_id"]
        bl = baseline_by_prompt.get(pid)
        is_baseline_row = (bl is row)

        # --- Token-level divergence vs baseline ---
        if is_baseline_row:
            row["exact_match"] = True
            row["first_divergence_token"] = -1
            row["common_prefix_tokens"] = int(row.get("output_tokens") or 0)
            row["common_prefix_frac"] = 1.0
            row["token_edit_distance"] = 0
            row["output_length_delta_tokens"] = 0
        else:
            policy_ids = row.get("output_token_ids") or []
            baseline_ids = (bl.get("output_token_ids") or []) if bl else []
            row.update(compute_divergence(policy_ids, baseline_ids))

        # --- Theoretical KV-cache bytes ---
        if can_score_kv:
            if is_baseline_row:
                # Baseline reference: no compression. Force the helper to
                # return ratio == 1.0 by claiming a window that covers the
                # entire sequence and an empty compressible-head selector.
                seq_len = int(row.get("prompt_tokens", 0)) + int(row.get("output_tokens", 0))
                kv = compute_kv_cache_bytes(
                    prompt_tokens=int(row.get("prompt_tokens", 0)),
                    output_tokens=int(row.get("output_tokens", 0)),
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    bit_width=16,
                    residual_window=max(seq_len, 1),
                    compressible_layers=(),
                    compressible_heads=(),
                    key_strategy="mse",
                    value_strategy="mse",
                )
            else:
                kv = compute_kv_cache_bytes(
                    prompt_tokens=int(row.get("prompt_tokens", 0)),
                    output_tokens=int(row.get("output_tokens", 0)),
                    num_layers=num_layers,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    bit_width=int(row.get("bit_width", 16)),
                    residual_window=int(row.get("residual_window", 0)),
                    compressible_layers=row.get("compressible_layers"),
                    compressible_heads=row.get("compressible_heads"),
                    key_strategy=str(row.get("key_strategy", "mse")),
                    value_strategy=str(row.get("value_strategy", "mse")),
                )
            row.update(kv)

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
    model_info: dict[str, int] | None = None,
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

    from .reporting import summarize_divergence
    divergence_summary = summarize_divergence(rows, baseline_policy_name)

    summary = {
        "study_name": study.name,
        "study_config": str(study_config_path) if study_config_path else "",
        "model_name": study.model.model_name,
        "policy_count": len(policies_used),
        "prompt_count": prompt_count,
        "row_count": len(rows),
        "repetitions": repetitions,
        "baseline_policy_name": baseline_policy_name,
        "divergence_summary": divergence_summary,
        # Persist model topology so a cold --rescore (no GPU, no torch
        # import) can rebuild KV-cache-byte annotations from rows.jsonl
        # alone.
        "model_info": model_info,
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
    # Captured once at first model load and threaded into score_results so
    # the post-hoc KV-cache-bytes calculation can run without holding a
    # torch reference (the model may already be torn down by the time
    # scoring runs in some control flows).
    model_info: dict[str, int] | None = None

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
            if model_info is None:
                cfg = model.config
                num_attn_heads = int(getattr(cfg, "num_attention_heads", 0))
                num_kv_heads = int(getattr(cfg, "num_key_value_heads", num_attn_heads))
                head_dim = int(
                    getattr(cfg, "head_dim", 0)
                    or (cfg.hidden_size // num_attn_heads if num_attn_heads else 0)
                )
                model_info = {
                    "num_hidden_layers": int(getattr(cfg, "num_hidden_layers", 0)),
                    "num_key_value_heads": num_kv_heads,
                    "head_dim": head_dim,
                }

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
    score_results(
        rows,
        baseline_policy_name=ctx.baseline_policy_name,
        model_info=model_info,
    )

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
        model_info=model_info,
    )

    # Final cleanup
    if model is not None:
        del model
        del tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return summary
