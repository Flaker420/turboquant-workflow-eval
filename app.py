"""Gradio UI for TurboQuant Workflow Eval."""

from __future__ import annotations

import csv
import gc
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from turboquant_workflow_eval.__main__ import (
    apply_set_policy_overrides,
    parse_int_list,
)
from turboquant_workflow_eval.download import (
    check_cache_status,
    discover_model_configs,
    download_one,
    format_summary_table,
)
from turboquant_workflow_eval.loader import (
    load_model_module,
    load_policy_module,
    load_study_module,
)
from turboquant_workflow_eval.prompts import filter_prompts, load_prompt_pack, load_prompt_source
from turboquant_workflow_eval.schema import (
    ModelConfig,
    PolicyConfig,
    PolicySettings,
    StudyConfig,
)
from turboquant_workflow_eval.scoring import _DEFAULT_THRESHOLDS

# ---------------------------------------------------------------------------
# Shared state for loaded model across tabs
# ---------------------------------------------------------------------------

_state: dict = {
    "model": None,
    "tokenizer": None,
    "loader_name": None,
    "model_cfg": None,
    "lm_root": None,
    "attention_blocks": None,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent


def _discover_py_configs(subdir: str) -> list[str]:
    root = _PROJECT_ROOT / "configs" / subdir
    if not root.exists():
        return []
    return sorted(
        str(p)
        for p in root.glob("*.py")
        if not p.name.startswith("_")
    )


def _discover_model_config_paths() -> list[str]:
    return _discover_py_configs("model")


def _discover_policy_config_paths() -> list[str]:
    return _discover_py_configs("policies")


def _discover_study_config_paths() -> list[str]:
    return _discover_py_configs("studies")


def _discover_output_dirs() -> list[str]:
    outputs_root = _PROJECT_ROOT / "outputs"
    if not outputs_root.exists():
        return []
    return sorted(
        str(d) for d in outputs_root.iterdir() if d.is_dir() and (d / "run_summary.json").exists()
    )


def _short_path(p: str) -> str:
    try:
        return str(Path(p).relative_to(_PROJECT_ROOT))
    except ValueError:
        return p


def _render_verdict_summary_html(summary: dict | None) -> str:
    """Render green/yellow/red verdict counts as colored HTML boxes."""
    if not summary:
        return ""
    counts = summary.get("verdict_summary", {})
    total = sum(counts.values()) or 1
    colors = {"green": "#2ecc71", "yellow": "#f39c12", "red": "#e74c3c"}
    boxes = []
    for level in ("green", "yellow", "red"):
        n = counts.get(level, 0)
        pct = n / total * 100
        boxes.append(
            f'<div style="display:inline-block;padding:12px 20px;margin:4px;'
            f"border-radius:8px;background:{colors[level]};color:white;"
            f'font-weight:bold;min-width:80px;text-align:center">'
            f"{level.upper()}<br>{n} ({pct:.0f}%)</div>"
        )
    return f'<div style="margin:8px 0">{"".join(boxes)}</div>'


def _style_verdict_column(csv_rows: list[dict]) -> str:
    """Render the CSV comparison table as an HTML table with colored verdict cells."""
    if not csv_rows:
        return ""
    colors = {"green": "#d5f5e3", "yellow": "#fdebd0", "red": "#fadbd8"}
    headers = list(csv_rows[0].keys())
    header_html = "".join(f"<th style='padding:6px 8px;border:1px solid #ddd'>{h}</th>" for h in headers)
    rows_html = []
    for row in csv_rows:
        cells = []
        for h in headers:
            val = row.get(h, "")
            style = "padding:6px 8px;border:1px solid #ddd"
            if h == "verdict" and val in colors:
                style += f";background:{colors[val]};font-weight:bold"
            cells.append(f"<td style='{style}'>{val}</td>")
        rows_html.append(f"<tr>{''.join(cells)}</tr>")
    return (
        "<div style='overflow-x:auto;max-height:500px;overflow-y:auto'>"
        f"<table style='border-collapse:collapse;font-size:13px'>"
        f"<thead><tr>{header_html}</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table></div>"
    )


# ---------------------------------------------------------------------------
# Tab 1 – Environment & Setup
# ---------------------------------------------------------------------------


def validate_environment():
    script = _PROJECT_ROOT / "scripts" / "validate_environment.py"
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(_PROJECT_ROOT),
            env={**__import__("os").environ, "PYTHONPATH": str(_PROJECT_ROOT / "src")},
        )
        output = result.stdout
        if result.stderr:
            output += "\n--- stderr ---\n" + result.stderr
        return output or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"


def check_model_cache(model_config_path):
    if not model_config_path:
        return "Select a model config first."
    model_cfg: ModelConfig = load_model_module(model_config_path)
    status = check_cache_status(model_cfg.model_name)
    lines = [
        f"Model: {model_cfg.model_name}",
        f"Model cached: {status['model_cached']}",
        f"Tokenizer cached: {status['tokenizer_cached']}",
    ]
    return "\n".join(lines)


def download_model(model_config_path):
    if not model_config_path:
        return "Select a model config first."
    from turboquant_workflow_eval.schema import model_to_legacy_dict

    model_cfg = load_model_module(model_config_path)
    result = download_one(model_to_legacy_dict(model_cfg))
    return format_summary_table([result])


def build_env_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Environment & Setup")

        with gr.Row():
            validate_btn = gr.Button("Validate Environment", variant="primary")
        validate_output = gr.Textbox(label="Validation Output", lines=12, interactive=False)
        validate_btn.click(fn=validate_environment, outputs=validate_output)

        gr.Markdown("---")
        gr.Markdown("### Model Cache")

        model_configs = _discover_model_config_paths()
        model_dd = gr.Dropdown(
            choices=model_configs,
            label="Model Config",
            info="Select a model config Python module",
        )
        with gr.Row():
            cache_btn = gr.Button("Check Cache")
            download_btn = gr.Button("Download Model")
        cache_output = gr.Textbox(label="Cache / Download Status", lines=6, interactive=False)

        cache_btn.click(fn=check_model_cache, inputs=model_dd, outputs=cache_output)
        download_btn.click(fn=download_model, inputs=model_dd, outputs=cache_output)

    return tab


# ---------------------------------------------------------------------------
# Tab 2 – Model Inspection
# ---------------------------------------------------------------------------


def load_model(model_config_path):
    if not model_config_path:
        return "Select a model config first.", None

    from turboquant_workflow_eval.model_loader import (
        infer_model_device,
        load_model_and_tokenizer,
        resolve_language_model_root,
    )
    from turboquant_workflow_eval.schema import model_to_legacy_dict

    model_cfg: ModelConfig = load_model_module(model_config_path)
    model, tokenizer, loader_name = load_model_and_tokenizer(
        model_to_legacy_dict(model_cfg)
    )
    lm_root = resolve_language_model_root(model)
    device = infer_model_device(model)

    _state.update(
        model=model,
        tokenizer=tokenizer,
        loader_name=loader_name,
        model_cfg=model_cfg,
        lm_root=lm_root,
        attention_blocks=None,
    )

    info = (
        f"Model: {model_cfg.model_name}\n"
        f"Loader: {loader_name}\n"
        f"Device: {device}\n"
        f"Dtype: {model_cfg.dtype}"
    )
    return info, None


def discover_blocks():
    if _state["model"] is None:
        return "Load a model first (use the Load Model button above).", None

    from turboquant_workflow_eval.module_discovery import discover_attention_blocks

    model_cfg: ModelConfig | None = _state["model_cfg"]
    expected = (
        model_cfg.layout.attention_blocks
        if model_cfg is not None and model_cfg.layout is not None
        else None
    )
    blocks = discover_attention_blocks(_state["lm_root"], expected_count=expected)
    _state["attention_blocks"] = blocks

    rows = []
    for b in blocks:
        rows.append([
            b.index,
            b.module_path,
            b.class_name,
            b.q_proj_path or "",
            b.k_proj_path or "",
            b.v_proj_path or "",
            b.packed_qkv_path or "",
            ", ".join(b.notes),
        ])

    headers = ["Index", "Module Path", "Class", "Q Proj", "K Proj", "V Proj", "Packed QKV", "Notes"]
    status = f"Discovered {len(blocks)} attention block(s)."
    return status, gr.Dataframe(value=rows, headers=headers)


def unload_model():
    """Free GPU memory by unloading the current model."""
    import torch

    if _state["model"] is None:
        return "No model loaded."
    _state.update(
        model=None, tokenizer=None, loader_name=None,
        model_cfg=None, lm_root=None, attention_blocks=None,
    )
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return "Model unloaded. GPU memory released."


def build_inspect_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Model Inspection")

        model_configs = _discover_model_config_paths()
        model_dd = gr.Dropdown(
            choices=model_configs,
            label="Model Config",
            info="Select a model config Python module",
        )
        with gr.Row():
            load_btn = gr.Button("Load Model", variant="primary")
            unload_btn = gr.Button("Unload Model", variant="stop")
        model_status = gr.Textbox(label="Model Status", lines=4, interactive=False)

        gr.Markdown("---")
        discover_btn = gr.Button("Discover Attention Blocks")
        block_status = gr.Textbox(label="Discovery Status", lines=1, interactive=False)
        block_table = gr.Dataframe(
            label="Attention Blocks",
            headers=["Index", "Module Path", "Class", "Q Proj", "K Proj", "V Proj", "Packed QKV", "Notes"],
            interactive=False,
        )

        load_btn.click(fn=load_model, inputs=model_dd, outputs=[model_status, block_table])
        unload_btn.click(fn=unload_model, outputs=model_status)
        discover_btn.click(fn=discover_blocks, outputs=[block_status, block_table])

    return tab


# ---------------------------------------------------------------------------
# Tab 3 – Preflight Stats
# ---------------------------------------------------------------------------


def run_preflight_ui(max_length, use_cache, progress=gr.Progress()):
    if _state["model"] is None:
        return "Load a model first (Tab 2).", None

    from turboquant_workflow_eval.module_discovery import discover_attention_blocks
    from turboquant_workflow_eval.preflight import run_preflight

    progress(0, desc="Preparing preflight...")

    if _state["attention_blocks"] is None:
        model_cfg: ModelConfig | None = _state["model_cfg"]
        expected = (
            model_cfg.layout.attention_blocks
            if model_cfg is not None and model_cfg.layout is not None
            else None
        )
        _state["attention_blocks"] = discover_attention_blocks(_state["lm_root"], expected_count=expected)

    progress(0.2, desc="Running preflight instrumentation...")
    prompts = load_prompt_source("builtin")
    report = run_preflight(
        model=_state["model"],
        tokenizer=_state["tokenizer"],
        language_model_root=_state["lm_root"],
        attention_blocks=_state["attention_blocks"],
        prompts=prompts,
        max_length=int(max_length),
        use_cache=bool(use_cache),
        loader_name=_state["loader_name"],
    )

    progress(1.0, desc="Done")
    status = (
        f"Preflight complete. {report['prompt_count']} prompts, "
        f"{len(report.get('attention_blocks', []))} blocks profiled."
    )
    return status, report


def build_preflight_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Preflight Stats")
        gr.Markdown("Requires a model loaded in the **Model Inspection** tab.")

        with gr.Row():
            max_length = gr.Number(label="Max Length", value=512, precision=0)
            use_cache = gr.Checkbox(label="Use Cache", value=True)

        run_btn = gr.Button("Run Preflight", variant="primary")
        preflight_status = gr.Textbox(label="Status", lines=2, interactive=False)
        preflight_json = gr.JSON(label="Preflight Report")

        run_btn.click(
            fn=run_preflight_ui,
            inputs=[max_length, use_cache],
            outputs=[preflight_status, preflight_json],
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 4 – Study Runner
# ---------------------------------------------------------------------------


def _default_policy_paths_for(study: StudyConfig) -> list[str]:
    """Pre-select the policy-module files whose POLICY.name matches this
    study's ``policies``.

    The new dataclass schema stores fully-loaded :class:`PolicyConfig`
    instances in ``study.policies`` rather than round-trippable paths, so
    we match by name against the discovered ``configs/policies/*.py``
    files. Any loose files that don't match are still offered in the
    checkbox as additional choices.
    """
    wanted = {p.name for p in study.policies}
    all_paths = _discover_policy_config_paths()
    selected: list[str] = []
    for path in all_paths:
        try:
            p = load_policy_module(path)
        except Exception:
            continue
        if p.name in wanted:
            selected.append(path)
    return selected


def load_study_policies(study_config_path):
    """Populate the Policy Configs checkbox with the study's defaults."""
    if not study_config_path:
        return gr.update(choices=[], value=[])
    try:
        study = load_study_module(study_config_path)
    except Exception:
        return gr.update(choices=[], value=[])
    defaults = _default_policy_paths_for(study)
    all_paths = _discover_policy_config_paths()
    # Offer every discovered policy module as a choice; pre-select the ones
    # this study already declares.
    return gr.update(choices=all_paths, value=defaults)


def load_prompt_pack_info(study_config_path):
    """Load and summarise the prompt pack(s) referenced by a study module."""
    if not study_config_path:
        return "", None
    try:
        study = load_study_module(study_config_path)
    except Exception as exc:
        return f"Error loading study config: {exc}", None

    prompts = []
    for pp_path in study.prompt_pack:
        try:
            prompts.extend(load_prompt_pack(pp_path))
        except FileNotFoundError:
            pass
        except Exception as exc:
            return f"Error loading prompt pack {pp_path}: {exc}", None

    categories = {}
    for p in prompts:
        categories[p.category] = categories.get(p.category, 0) + 1
    cat_summary = ", ".join(f"{k}: {v}" for k, v in sorted(categories.items()))
    summary = f"{len(prompts)} prompts — {cat_summary}"

    rows = []
    for p in prompts:
        rows.append([
            p.id,
            p.category,
            p.title,
            "yes" if p.reference_answer else "",
            "yes" if p.test_cases else "",
            (p.prompt[:100] + "...") if len(p.prompt) > 100 else p.prompt,
        ])
    return summary, rows


_study_controller = None


def run_study_ui(study_config_path, policy_paths, output_dir,
                 repetitions, temperature, max_new_tokens, shuffle_policies,
                 compressible_layers_text,
                 compressible_heads_text,
                 model_config_override,
                 prompt_category_filter, prompt_id_filter,
                 policy_overrides_text="",
                 progress=gr.Progress()):
    global _study_controller
    if not study_config_path:
        return "Select a study config.", None, ""
    if not policy_paths:
        return "Select at least one policy.", None, ""

    from turboquant_workflow_eval.events import EventBus
    from turboquant_workflow_eval.study import StudyController, run_workflow_study

    output_dir = output_dir.strip() or "outputs/study_run"

    # --- Load the study as a dataclass and apply UI overrides on top ----
    try:
        study: StudyConfig = load_study_module(study_config_path)
    except Exception as exc:
        return f"Failed to load study config: {exc}", None, ""

    # 1. --model override: swap the entire model field.
    if model_config_override:
        try:
            study = replace(study, model=load_model_module(model_config_override))
        except Exception as exc:
            return f"Failed to load model override: {exc}", None, ""

    # 2. --policies override: rebuild study.policies from the checkbox selection.
    try:
        new_policies = tuple(load_policy_module(p) for p in policy_paths)
    except Exception as exc:
        return f"Failed to load a selected policy: {exc}", None, ""
    # Preserve baseline_policy_name if still represented; otherwise reset to
    # the first selected policy (StudyConfig's __post_init__ enforces this
    # invariant for multi-policy studies).
    new_names = {p.name for p in new_policies}
    new_baseline = (
        study.baseline_policy_name
        if study.baseline_policy_name in new_names
        else new_policies[0].name
    )
    study = replace(study, policies=new_policies, baseline_policy_name=new_baseline)

    # 3. Runtime overrides from the accordion widgets.
    runtime_updates: dict = {}
    if repetitions is not None and repetitions > 0:
        runtime_updates["repetitions"] = int(repetitions)
    if temperature is not None and temperature >= 0:
        runtime_updates["temperature"] = float(temperature)
    if max_new_tokens is not None and max_new_tokens > 0:
        runtime_updates["max_new_tokens"] = int(max_new_tokens)
    if shuffle_policies:
        runtime_updates["shuffle_policies"] = True
    if runtime_updates:
        study = replace(study, runtime=replace(study.runtime, **runtime_updates))

    # 4. Global compressible_layers textbox — applies to every policy.
    cl_text = (compressible_layers_text or "").strip()
    if cl_text:
        try:
            cl = parse_int_list(cl_text)
        except Exception as exc:
            return f"Invalid compressible_layers {cl_text!r}: {exc}", None, ""
        new_pols = tuple(
            replace(p, settings=replace(p.settings, compressible_layers=cl))
            for p in study.policies
        )
        study = replace(study, policies=new_pols)

    # 4b. Global compressible_heads textbox — applies to every policy.
    ch_text = (compressible_heads_text or "").strip()
    if ch_text:
        try:
            ch = parse_int_list(ch_text)
        except Exception as exc:
            return f"Invalid compressible_heads {ch_text!r}: {exc}", None, ""
        new_pols = tuple(
            replace(p, settings=replace(p.settings, compressible_heads=ch))
            for p in study.policies
        )
        study = replace(study, policies=new_pols)

    # 5. Free-form policy overrides textbox (one NAME.DOT.KEY=VALUE per line).
    override_lines = [
        line.strip()
        for line in (policy_overrides_text or "").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if override_lines:
        try:
            study = apply_set_policy_overrides(study, override_lines)
        except SystemExit as exc:  # helper raises SystemExit for CLI parity
            return f"Policy override error: {exc}", None, ""
        except Exception as exc:
            return f"Policy override error: {exc}", None, ""

    # --- Prompt filtering (orthogonal to the dataclass) ----
    prompt_cats = [c.strip() for c in (prompt_category_filter or "").split(",") if c.strip()] or None
    prompt_ids = [i.strip() for i in (prompt_id_filter or "").split(",") if i.strip()] or None

    # --- Event bus + controller for live tracking ----
    event_bus = EventBus()
    _study_controller = StudyController(event_bus=event_bus)
    completed_prompts = []

    def _on_event(event):
        if event.kind == "prompt_completed":
            completed_prompts.append(event.data)

    event_bus.subscribe(_on_event)

    progress(0, desc="Starting study...")

    def _progress_cb(frac: float, desc: str = "") -> None:
        progress(frac, desc=desc)

    try:
        summary = run_workflow_study(
            study,
            output_dir=output_dir,
            progress_callback=_progress_cb,
            prompt_ids=prompt_ids,
            prompt_categories=prompt_cats,
            event_bus=event_bus,
            controller=_study_controller,
            study_config_path=study_config_path,
        )
        progress(1.0, desc="Complete")

        verdict_html = _render_verdict_summary_html(summary)
        status = (
            f"Study complete: {summary['row_count']} rows across "
            f"{summary['policy_count']} policy(ies), "
            f"{summary['prompt_count']} prompt(s).\n"
            f"Outputs written to: {summary['output_dir']}"
        )
        return status, summary, verdict_html
    except Exception as exc:
        return f"Study failed: {exc}", None, ""
    finally:
        _study_controller = None


def stop_study():
    """Signal the running study to stop."""
    global _study_controller
    if _study_controller:
        _study_controller.stop()
        return "Stop requested. Study will halt after current prompt."
    return "No study is running."


def pause_study():
    """Toggle pause on the running study."""
    global _study_controller
    if _study_controller:
        if _study_controller.paused:
            _study_controller.resume()
            return "Resumed."
        else:
            _study_controller.pause()
            return "Paused. Click again to resume."
    return "No study is running."


def build_study_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Study Runner")

        study_configs = _discover_study_config_paths()
        default_study = study_configs[0] if study_configs else None
        all_policy_paths = _discover_policy_config_paths()
        default_policies: list[str] = []
        if default_study:
            try:
                _study = load_study_module(default_study)
                default_policies = _default_policy_paths_for(_study)
            except Exception:
                pass

        study_dd = gr.Dropdown(
            choices=study_configs,
            value=default_study,
            label="Study Config",
            info="Select a study config Python module",
        )

        policy_cb = gr.CheckboxGroup(
            choices=all_policy_paths,
            value=default_policies,
            label="Policy Configs",
            info="Policies to evaluate (auto-populated from study config)",
        )
        study_dd.change(fn=load_study_policies, inputs=study_dd, outputs=policy_cb)

        output_dir = gr.Textbox(
            label="Output Directory",
            value="outputs/study_run",
            info="Where to write study results",
        )

        model_configs = _discover_model_config_paths()
        model_override_dd = gr.Dropdown(
            choices=[""] + model_configs,
            value="",
            label="Model Config Override",
            info="Leave blank to use the model from the study config, or select to override",
        )

        with gr.Accordion("Runtime Overrides", open=False):
            gr.Markdown("Leave blank to use defaults from the study config.")
            with gr.Row():
                repetitions = gr.Number(label="Repetitions", precision=0)
                temperature = gr.Number(label="Temperature")
                max_new_tokens = gr.Number(label="Max New Tokens", precision=0)
            shuffle_policies = gr.Checkbox(label="Shuffle policy order", value=False)
            compressible_layers_box = gr.Textbox(
                label="Compressible layers",
                placeholder="e.g. 7,15,23,31 (blank = backend default)",
                info=(
                    "Comma-separated layer indices applied to every policy's "
                    "PolicySettings.compressible_layers. Blank uses the "
                    "backend default (Qwen3.5: every 4th layer; dense backends: "
                    "all layers). For per-policy targeting, use the Policy "
                    "Overrides accordion below."
                ),
            )
            compressible_heads_box = gr.Textbox(
                label="Compressible heads",
                placeholder="e.g. 0,2 (blank = all heads)",
                info=(
                    "Comma-separated KV-head indices applied to every policy's "
                    "PolicySettings.compressible_heads. Blank compresses every "
                    "head. Heads are validated per-backend against num_kv_heads "
                    "at prepare_model time."
                ),
            )

        with gr.Accordion("Policy Overrides", open=False):
            gr.Markdown(
                "Override fields on a specific policy at run time. One override per line. "
                "Format: `<policy_name|*>.<dot.key>=<value>`. The first segment matches "
                "`PolicyConfig.name`; `*` matches every policy. Overrides are applied via "
                "`schema.replace_path`, so `__post_init__` re-validates the result and any "
                "invalid value fails fast.\n\n"
                "Examples:\n"
                "```\n"
                "turboquant_safe.settings.key_strategy=mse\n"
                "*.settings.bit_width=8\n"
                "turboquant_safe.settings.compressible_layers=7,15,23,31\n"
                "baseline.enabled=false\n"
                "```"
            )
            policy_overrides_box = gr.Textbox(
                label="Policy overrides",
                lines=4,
                placeholder="turboquant_safe.settings.key_strategy=mse",
            )

        with gr.Accordion("Prompt Filtering", open=False):
            gr.Markdown("Filter which prompts to run. Leave blank for all prompts.")
            prompt_category_filter = gr.Textbox(
                label="Category Filter",
                placeholder="e.g. math,coding (comma-separated)",
            )
            prompt_id_filter = gr.Textbox(
                label="Prompt ID Filter",
                placeholder="e.g. math_01,coding_02 (comma-separated)",
            )

        default_prompt_summary = ""
        default_prompt_rows = None
        if default_study:
            try:
                default_prompt_summary, default_prompt_rows = load_prompt_pack_info(default_study)
            except Exception:
                pass

        with gr.Accordion("Prompt Pack Preview", open=False):
            prompt_summary = gr.Textbox(
                label="Summary", lines=1, interactive=False,
                value=default_prompt_summary,
            )
            prompt_table = gr.Dataframe(
                label="Prompts",
                headers=["ID", "Category", "Title", "Reference", "Tests", "Preview"],
                interactive=False,
                value=default_prompt_rows,
            )
            study_dd.change(
                fn=load_prompt_pack_info, inputs=study_dd,
                outputs=[prompt_summary, prompt_table],
            )

        with gr.Row():
            run_btn = gr.Button("Run Study", variant="primary")
            pause_btn = gr.Button("Pause / Resume")
            stop_btn = gr.Button("Stop", variant="stop")

        study_status = gr.Textbox(label="Status", lines=4, interactive=False)

        gr.Markdown("### Live Verdict Summary")
        live_verdict_html = gr.HTML()

        study_json = gr.JSON(label="Run Summary")

        run_btn.click(
            fn=run_study_ui,
            inputs=[study_dd, policy_cb, output_dir,
                    repetitions, temperature, max_new_tokens, shuffle_policies,
                    compressible_layers_box,
                    compressible_heads_box,
                    model_override_dd,
                    prompt_category_filter, prompt_id_filter,
                    policy_overrides_box],
            outputs=[study_status, study_json, live_verdict_html],
        )
        pause_btn.click(fn=pause_study, outputs=study_status)
        stop_btn.click(fn=stop_study, outputs=study_status)

    return tab


# ---------------------------------------------------------------------------
# Tab 5 – Results Explorer
# ---------------------------------------------------------------------------


def load_results(output_dir_path):
    if not output_dir_path:
        return "Select an output directory.", "", "", gr.update(choices=[], value=None), "", "", None

    out = Path(output_dir_path)

    # Load CSV rows for styled table
    csv_path = out / "workflow_compare.csv"
    csv_rows: list[dict] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
    styled_table_html = _style_verdict_column(csv_rows)

    # Load summary
    summary_path = out / "run_summary.json"
    summary = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    verdict_html = _render_verdict_summary_html(summary)

    # Load examples.md
    examples_path = out / "examples.md"
    examples_md = ""
    if examples_path.exists():
        examples_md = examples_path.read_text(encoding="utf-8")

    # Discover text outputs for prompt selector
    text_dir = out / "text_outputs"
    prompt_files = []
    if text_dir.exists():
        prompt_files = sorted(str(p.name) for p in text_dir.glob("*.md"))

    status = f"Loaded results from {_short_path(output_dir_path)}."
    return (
        status,
        verdict_html,
        styled_table_html,
        gr.update(choices=prompt_files, value=prompt_files[0] if prompt_files else None),
        "",
        examples_md,
        summary,
    )


def view_prompt_output(output_dir_path, prompt_file):
    if not output_dir_path or not prompt_file:
        return ""
    path = Path(output_dir_path) / "text_outputs" / prompt_file
    if path.exists():
        return path.read_text(encoding="utf-8")
    return "(file not found)"


def refresh_output_dirs():
    dirs = _discover_output_dirs()
    return gr.update(choices=dirs, value=dirs[0] if dirs else None)


def build_results_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Results Explorer")

        with gr.Row():
            output_dirs = _discover_output_dirs()
            output_dd = gr.Dropdown(
                choices=output_dirs,
                label="Output Directory",
                info="Select a completed study output",
            )
            refresh_btn = gr.Button("Refresh", scale=0)

        refresh_btn.click(fn=refresh_output_dirs, outputs=output_dd)

        load_btn = gr.Button("Load Results", variant="primary")
        results_status = gr.Textbox(label="Status", lines=1, interactive=False)

        gr.Markdown("### Verdict Summary")
        verdict_html = gr.HTML()

        gr.Markdown("### Comparison Table")
        results_table = gr.HTML(label="workflow_compare.csv")

        gr.Markdown("### Prompt Outputs")
        prompt_dd = gr.Dropdown(choices=[], label="Select Prompt Output")
        prompt_md = gr.Markdown(label="Output")

        with gr.Accordion("Side-by-Side Comparison (examples.md)", open=False):
            examples_md = gr.Markdown()

        gr.Markdown("### Run Summary")
        summary_json = gr.JSON(label="run_summary.json")

        load_btn.click(
            fn=load_results,
            inputs=output_dd,
            outputs=[results_status, verdict_html, results_table,
                     prompt_dd, prompt_md, examples_md, summary_json],
        )
        prompt_dd.change(
            fn=view_prompt_output,
            inputs=[output_dd, prompt_dd],
            outputs=prompt_md,
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 6 – Quick Test (single prompt, instant feedback)
# ---------------------------------------------------------------------------


def _load_prompt_choices():
    """Load prompts from every discovered study module for the quick-test dropdown."""
    prompts = []
    for study_path in (_PROJECT_ROOT / "configs" / "studies").glob("*.py"):
        if study_path.name.startswith("_"):
            continue
        try:
            study = load_study_module(study_path)
        except Exception:
            continue
        for pp_path in study.prompt_pack:
            try:
                prompts.extend(load_prompt_pack(pp_path))
            except Exception:
                continue
    # Deduplicate by id
    seen = set()
    unique = []
    for p in prompts:
        if p.id not in seen:
            seen.add(p.id)
            unique.append(p)
    return unique


_QUICK_TEST_PROMPTS = _load_prompt_choices()
_PROMPT_MAP = {f"{p.id} — {p.title}": p for p in _QUICK_TEST_PROMPTS}


def run_quick_test(
    policy_path, prompt_choice, custom_prompt, max_new_tokens, temperature, repetitions,
):
    """Run a single prompt with a single policy, return instant results."""
    if _state["model"] is None:
        return "Load a model first (Tab 2: Model Inspection).", "", None

    from turboquant_workflow_eval.generation import generate_one
    from turboquant_workflow_eval.import_utils import load_object
    from turboquant_workflow_eval.schema import (
        model_to_legacy_dict,
        policy_to_legacy_dict,
    )
    from turboquant_workflow_eval.scoring import check_reference_answer, compute_verdict

    model = _state["model"]
    tokenizer = _state["tokenizer"]
    model_cfg: ModelConfig | None = _state["model_cfg"]

    # Resolve prompt
    prompt_text = custom_prompt.strip() if custom_prompt.strip() else None
    prompt_spec = None
    if not prompt_text and prompt_choice:
        prompt_spec = _PROMPT_MAP.get(prompt_choice)
        if prompt_spec:
            prompt_text = prompt_spec.prompt

    if not prompt_text:
        return "Enter a prompt or select one from the dropdown.", "", None

    # Load and apply adapter. When no policy is selected we fall back to the
    # pass-through baseline policy (no quantization) so the Quick Test tab
    # can run without forcing the user to pick one.
    adapter = None
    if policy_path:
        try:
            policy = load_policy_module(policy_path)
        except Exception as exc:
            return f"Failed to load policy: {exc}", "", None
        try:
            adapter_cls = load_object(policy.adapter.import_path)
            adapter = adapter_cls()
            if model_cfg is None:
                return "Model config is missing from shared state.", "", None
            model, tokenizer = adapter.prepare_model(
                model,
                tokenizer,
                model_to_legacy_dict(model_cfg),
                policy_to_legacy_dict(policy),
            )
        except Exception as exc:
            return f"Adapter failed: {exc}", "", None

    # Build runtime config
    runtime_cfg = {
        "max_input_tokens": 4096,
        "max_new_tokens": int(max_new_tokens) if max_new_tokens else 256,
        "do_sample": float(temperature or 0) > 0,
        "temperature": float(temperature) if temperature else 0.0,
    }

    # Run
    try:
        reps = max(1, int(repetitions or 1))
        results = []
        for _ in range(reps):
            result = generate_one(model, tokenizer, prompt_text, runtime_cfg)
            results.append(result)

        main_result = results[0]
        output_text = main_result["output_text"]

        # Metrics
        avg_latency = sum(r["latency_s"] for r in results) / len(results)
        tps = main_result.get("tokens_per_second")
        vram = main_result.get("peak_vram_gb")

        metrics = {
            "latency_s": round(avg_latency, 4),
            "tokens_per_second": round(tps, 2) if tps else None,
            "peak_vram_gb": round(vram, 3) if vram else None,
            "output_tokens": main_result.get("output_tokens", 0),
            "prompt_tokens": main_result.get("prompt_tokens", 0),
            "repetitions": reps,
        }

        # Scoring
        if prompt_spec and prompt_spec.reference_answer:
            metrics["math_correct"] = check_reference_answer(output_text, prompt_spec.reference_answer)

        status = f"Completed in {avg_latency:.2f}s ({metrics['output_tokens']} tokens)"
    except Exception as exc:
        output_text = f"[ERROR] {exc}"
        metrics = {"error": str(exc)}
        status = f"Failed: {exc}"
    finally:
        # Revert adapter
        if adapter:
            if adapter.can_revert():
                adapter.revert(model)
            else:
                adapter.cleanup(model)

    return status, output_text, metrics


def build_quick_test_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Quick Test")
        gr.Markdown("Run a single prompt with instant feedback. Requires a model loaded in Tab 2.")

        with gr.Row():
            policy_configs = _discover_policy_config_paths()
            policy_dd = gr.Dropdown(
                choices=[""] + policy_configs,
                value="",
                label="Policy (optional)",
                info="Leave blank for no compression (raw model)",
            )

        prompt_choices = list(_PROMPT_MAP.keys())
        prompt_dd = gr.Dropdown(
            choices=[""] + prompt_choices,
            value="",
            label="Select Prompt (from prompt pack)",
        )
        custom_prompt = gr.Textbox(
            label="Or enter custom prompt",
            placeholder="Type your own prompt here...",
            lines=4,
        )

        with gr.Row():
            max_new_tokens = gr.Number(label="Max New Tokens", value=256, precision=0)
            temperature = gr.Number(label="Temperature", value=0.0)
            repetitions = gr.Number(label="Repetitions", value=1, precision=0)

        run_btn = gr.Button("Run Quick Test", variant="primary")
        status_box = gr.Textbox(label="Status", lines=1, interactive=False)

        gr.Markdown("### Output")
        output_box = gr.Textbox(label="Model Output", lines=12, interactive=False)

        gr.Markdown("### Metrics")
        metrics_json = gr.JSON(label="Metrics")

        run_btn.click(
            fn=run_quick_test,
            inputs=[policy_dd, prompt_dd, custom_prompt, max_new_tokens, temperature, repetitions],
            outputs=[status_box, output_box, metrics_json],
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 7 – Re-Score Results
# ---------------------------------------------------------------------------


def rescore_ui(
    output_dir_path,
    latency_yellow, latency_red,
    similarity_yellow, similarity_red,
    length_yellow, length_red,
):
    """Re-score existing results with new thresholds."""
    if not output_dir_path:
        return "Select an output directory.", "", None

    from turboquant_workflow_eval.rescoring import rescore

    thresholds = {
        "latency_yellow_pct": float(latency_yellow),
        "latency_red_pct": float(latency_red),
        "similarity_yellow": float(similarity_yellow),
        "similarity_red": float(similarity_red),
        "output_length_yellow_pct": float(length_yellow),
        "output_length_red_pct": float(length_red),
    }

    rows_path = Path(output_dir_path) / "rows.jsonl"
    if not rows_path.exists():
        return f"No rows.jsonl found in {output_dir_path}", "", None

    try:
        rows = rescore(
            rows_jsonl_path=rows_path,
            thresholds=thresholds,
            output_dir=output_dir_path,
        )
    except Exception as exc:
        return f"Re-scoring failed: {exc}", "", None

    # Build verdict summary
    verdict_counts = {"green": 0, "yellow": 0, "red": 0}
    for row in rows:
        v = row.get("verdict", "green")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    summary = {"verdict_summary": verdict_counts, "row_count": len(rows)}
    verdict_html = _render_verdict_summary_html(summary)

    # Reload CSV for table
    csv_path = Path(output_dir_path) / "workflow_compare.csv"
    csv_rows = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8") as f:
            csv_rows = list(csv.DictReader(f))
    styled = _style_verdict_column(csv_rows)

    status = f"Re-scored {len(rows)} rows with new thresholds."
    return status, verdict_html, styled


def build_rescore_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Re-Score Results")
        gr.Markdown("Adjust verdict thresholds and instantly re-score existing results. **No GPU needed.**")

        output_dirs = _discover_output_dirs()
        output_dd = gr.Dropdown(
            choices=output_dirs,
            label="Output Directory",
            info="Select a completed study output",
        )

        gr.Markdown("### Threshold Controls")
        with gr.Row():
            latency_yellow = gr.Slider(0, 100, value=_DEFAULT_THRESHOLDS["latency_yellow_pct"], step=1, label="Latency Yellow %")
            latency_red = gr.Slider(0, 100, value=_DEFAULT_THRESHOLDS["latency_red_pct"], step=1, label="Latency Red %")
        with gr.Row():
            similarity_yellow = gr.Slider(0, 1, value=_DEFAULT_THRESHOLDS["similarity_yellow"], step=0.01, label="Similarity Yellow")
            similarity_red = gr.Slider(0, 1, value=_DEFAULT_THRESHOLDS["similarity_red"], step=0.01, label="Similarity Red")
        with gr.Row():
            length_yellow = gr.Slider(0, 100, value=_DEFAULT_THRESHOLDS["output_length_yellow_pct"], step=1, label="Output Length Yellow %")
            length_red = gr.Slider(0, 100, value=_DEFAULT_THRESHOLDS["output_length_red_pct"], step=1, label="Output Length Red %")

        rescore_btn = gr.Button("Re-Score", variant="primary")
        rescore_status = gr.Textbox(label="Status", lines=1, interactive=False)

        gr.Markdown("### Updated Verdicts")
        verdict_html = gr.HTML()
        results_table = gr.HTML(label="Updated Comparison Table")

        rescore_btn.click(
            fn=rescore_ui,
            inputs=[output_dd, latency_yellow, latency_red, similarity_yellow, similarity_red, length_yellow, length_red],
            outputs=[rescore_status, verdict_html, results_table],
        )

    return tab


# ---------------------------------------------------------------------------
# Tab 8 – Side-by-Side Comparison
# ---------------------------------------------------------------------------


def compare_runs(dir_a, dir_b):
    """Load two run outputs and compare side-by-side."""
    if not dir_a or not dir_b:
        return "Select two output directories.", ""

    def _load_rows(d):
        path = Path(d) / "rows.jsonl"
        rows = {}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        rows[(row["policy_name"], row["prompt_id"])] = row
        return rows

    rows_a = _load_rows(dir_a)
    rows_b = _load_rows(dir_b)

    # Build comparison for matching prompt_ids
    all_keys = sorted(set(rows_a.keys()) | set(rows_b.keys()))
    if not all_keys:
        return "No rows found in either directory.", ""

    html_parts = [
        "<style>",
        ".cmp-table { border-collapse: collapse; font-size: 13px; width: 100%; }",
        ".cmp-table th, .cmp-table td { padding: 6px 8px; border: 1px solid #ddd; }",
        ".cmp-table th { background: #f5f5f5; }",
        ".diff { background: #fff3cd; }",
        ".verdict-green { color: #2ecc71; font-weight: bold; }",
        ".verdict-yellow { color: #f39c12; font-weight: bold; }",
        ".verdict-red { color: #e74c3c; font-weight: bold; }",
        "</style>",
    ]

    label_a = _short_path(dir_a)
    label_b = _short_path(dir_b)

    html_parts.append(f"<h3>Comparing: {label_a} vs {label_b}</h3>")
    html_parts.append(
        "<table class='cmp-table'>"
        "<thead><tr>"
        "<th>Policy</th><th>Prompt</th>"
        f"<th>Verdict ({label_a})</th><th>Verdict ({label_b})</th>"
        f"<th>Latency ({label_a})</th><th>Latency ({label_b})</th>"
        f"<th>Tokens ({label_a})</th><th>Tokens ({label_b})</th>"
        "</tr></thead><tbody>"
    )

    for key in all_keys:
        ra = rows_a.get(key, {})
        rb = rows_b.get(key, {})
        va = ra.get("verdict", "—")
        vb = rb.get("verdict", "—")
        diff_class = " class='diff'" if va != vb else ""

        def _fmt_verdict(v):
            return f"<span class='verdict-{v}'>{v.upper()}</span>" if v in ("green", "yellow", "red") else v

        html_parts.append(
            f"<tr{diff_class}>"
            f"<td>{key[0]}</td><td>{key[1]}</td>"
            f"<td>{_fmt_verdict(va)}</td><td>{_fmt_verdict(vb)}</td>"
            f"<td>{ra.get('latency_s', '—'):.4f}</td><td>{rb.get('latency_s', '—') if isinstance(rb.get('latency_s'), (int, float)) else '—'}</td>"
            f"<td>{ra.get('output_tokens', '—')}</td><td>{rb.get('output_tokens', '—')}</td>"
            f"</tr>"
        )

    html_parts.append("</tbody></table>")

    changed = sum(1 for k in all_keys if rows_a.get(k, {}).get("verdict") != rows_b.get(k, {}).get("verdict"))
    status = f"Compared {len(all_keys)} rows, {changed} verdict(s) differ."
    return status, "\n".join(html_parts)


def build_comparison_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Side-by-Side Comparison")
        gr.Markdown("Compare two study runs to see what changed.")

        output_dirs = _discover_output_dirs()
        with gr.Row():
            dir_a = gr.Dropdown(choices=output_dirs, label="Run A")
            dir_b = gr.Dropdown(choices=output_dirs, label="Run B")

        compare_btn = gr.Button("Compare", variant="primary")
        compare_status = gr.Textbox(label="Status", lines=1, interactive=False)
        comparison_html = gr.HTML(label="Comparison")

        compare_btn.click(
            fn=compare_runs,
            inputs=[dir_a, dir_b],
            outputs=[compare_status, comparison_html],
        )

    return tab


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    env_tab = build_env_tab()
    inspect_tab = build_inspect_tab()
    preflight_tab = build_preflight_tab()
    study_tab = build_study_tab()
    results_tab = build_results_tab()
    quick_test_tab = build_quick_test_tab()
    rescore_tab = build_rescore_tab()
    comparison_tab = build_comparison_tab()

    demo = gr.TabbedInterface(
        [env_tab, inspect_tab, preflight_tab, study_tab, results_tab,
         quick_test_tab, rescore_tab, comparison_tab],
        ["Environment", "Model Inspection", "Preflight", "Study Runner", "Results",
         "Quick Test", "Re-Score", "Comparison"],
        title="TurboQuant Workflow Eval",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
