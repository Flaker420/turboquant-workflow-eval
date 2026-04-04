"""Gradio UI for TurboQuant Workflow Eval."""

from __future__ import annotations

import csv
import gc
import json
import subprocess
import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from turboquant_workflow_eval.config import load_yaml, resolve_relative_path
from turboquant_workflow_eval.download import (
    check_cache_status,
    discover_model_configs,
    download_one,
    format_summary_table,
)
from turboquant_workflow_eval.prompts import load_prompt_pack, load_prompt_source

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


def _discover_model_config_paths() -> list[str]:
    return sorted(str(p) for p in (_PROJECT_ROOT / "configs" / "model").glob("*.yaml"))


def _discover_policy_config_paths() -> list[str]:
    return sorted(str(p) for p in (_PROJECT_ROOT / "configs" / "policies").glob("*.yaml"))


def _discover_study_config_paths() -> list[str]:
    return sorted(str(p) for p in (_PROJECT_ROOT / "configs" / "studies").glob("*.yaml"))


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
    cfg = load_yaml(model_config_path)
    status = check_cache_status(cfg["model_name"])
    lines = [
        f"Model: {cfg['model_name']}",
        f"Model cached: {status['model_cached']}",
        f"Tokenizer cached: {status['tokenizer_cached']}",
    ]
    return "\n".join(lines)


def download_model(model_config_path):
    if not model_config_path:
        return "Select a model config first."
    cfg = load_yaml(model_config_path)
    result = download_one(cfg)
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
            info="Select a model config YAML",
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

    cfg = load_yaml(model_config_path)
    model, tokenizer, loader_name = load_model_and_tokenizer(cfg)
    lm_root = resolve_language_model_root(model)
    device = infer_model_device(model)

    _state.update(
        model=model,
        tokenizer=tokenizer,
        loader_name=loader_name,
        model_cfg=cfg,
        lm_root=lm_root,
        attention_blocks=None,
    )

    info = (
        f"Model: {cfg['model_name']}\n"
        f"Loader: {loader_name}\n"
        f"Device: {device}\n"
        f"Dtype: {cfg.get('dtype', 'unknown')}"
    )
    return info, None


def discover_blocks():
    if _state["model"] is None:
        return "Load a model first (use the Load Model button above).", None

    from turboquant_workflow_eval.module_discovery import discover_attention_blocks

    cfg = _state["model_cfg"] or {}
    expected = cfg.get("layout", {}).get("attention_blocks")
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
            info="Select a model config YAML",
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
        cfg = _state["model_cfg"] or {}
        expected = cfg.get("layout", {}).get("attention_blocks")
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


def load_study_policies(study_config_path):
    """Return the policy config paths referenced by the study config."""
    if not study_config_path:
        return gr.update(choices=[], value=[])
    cfg = load_yaml(study_config_path)
    policies = []
    for item in cfg.get("policy_configs", []):
        resolved = resolve_relative_path(study_config_path, item)
        policies.append(str(resolved))
    return gr.update(choices=policies, value=policies)


def load_prompt_pack_info(study_config_path):
    """Load and summarise the prompt pack referenced by a study config."""
    if not study_config_path:
        return "", None
    cfg = load_yaml(study_config_path)
    prompt_pack_cfg = cfg["prompt_pack"]
    if isinstance(prompt_pack_cfg, list):
        prompts = []
        for pp in prompt_pack_cfg:
            prompts.extend(load_prompt_pack(resolve_relative_path(study_config_path, pp)))
    else:
        prompts = load_prompt_pack(resolve_relative_path(study_config_path, prompt_pack_cfg))

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


def run_study_ui(study_config_path, policy_paths, output_dir,
                 repetitions, temperature, max_new_tokens, shuffle_policies,
                 progress=gr.Progress()):
    if not study_config_path:
        return "Select a study config.", None
    if not policy_paths:
        return "Select at least one policy.", None

    from turboquant_workflow_eval.study import run_workflow_study

    output_dir = output_dir.strip() or "outputs/study_run"
    policy_arg = ",".join(policy_paths)

    # Build runtime overrides from non-default UI values
    overrides: dict = {}
    if repetitions is not None and repetitions > 0:
        overrides["repetitions"] = int(repetitions)
    if temperature is not None and temperature >= 0:
        overrides["temperature"] = float(temperature)
    if max_new_tokens is not None and max_new_tokens > 0:
        overrides["max_new_tokens"] = int(max_new_tokens)
    if shuffle_policies:
        overrides["shuffle_policies"] = True

    progress(0, desc="Starting study...")

    def _progress_cb(frac: float, desc: str = "") -> None:
        progress(frac, desc=desc)

    try:
        summary = run_workflow_study(
            study_config_path=study_config_path,
            output_dir=output_dir,
            policy_configs_arg=policy_arg,
            runtime_overrides=overrides or None,
            progress_callback=_progress_cb,
        )
        progress(1.0, desc="Complete")
        status = (
            f"Study complete: {summary['row_count']} rows across "
            f"{summary['policy_count']} policy(ies), "
            f"{summary['prompt_count']} prompt(s).\n"
            f"Outputs written to: {summary['output_dir']}"
        )
        return status, summary
    except Exception as exc:
        return f"Study failed: {exc}", None


def build_study_tab():
    with gr.Blocks() as tab:
        gr.Markdown("## Study Runner")

        study_configs = _discover_study_config_paths()
        study_dd = gr.Dropdown(
            choices=study_configs,
            label="Study Config",
            info="Select a study config YAML",
        )

        policy_cb = gr.CheckboxGroup(
            choices=[],
            label="Policy Configs",
            info="Policies to evaluate (auto-populated from study config)",
        )
        study_dd.change(fn=load_study_policies, inputs=study_dd, outputs=policy_cb)

        output_dir = gr.Textbox(
            label="Output Directory",
            value="outputs/study_run",
            info="Where to write study results",
        )

        with gr.Accordion("Runtime Overrides", open=False):
            gr.Markdown("Leave blank to use defaults from the study config.")
            with gr.Row():
                repetitions = gr.Number(label="Repetitions", precision=0)
                temperature = gr.Number(label="Temperature")
                max_new_tokens = gr.Number(label="Max New Tokens", precision=0)
            shuffle_policies = gr.Checkbox(label="Shuffle policy order", value=False)

        with gr.Accordion("Prompt Pack Preview", open=False):
            prompt_summary = gr.Textbox(label="Summary", lines=1, interactive=False)
            prompt_table = gr.Dataframe(
                label="Prompts",
                headers=["ID", "Category", "Title", "Reference", "Tests", "Preview"],
                interactive=False,
            )
            study_dd.change(
                fn=load_prompt_pack_info, inputs=study_dd,
                outputs=[prompt_summary, prompt_table],
            )

        run_btn = gr.Button("Run Study", variant="primary")
        study_status = gr.Textbox(label="Status", lines=4, interactive=False)
        study_json = gr.JSON(label="Run Summary")

        run_btn.click(
            fn=run_study_ui,
            inputs=[study_dd, policy_cb, output_dir,
                    repetitions, temperature, max_new_tokens, shuffle_policies],
            outputs=[study_status, study_json],
        )

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
# Main
# ---------------------------------------------------------------------------


def main():
    env_tab = build_env_tab()
    inspect_tab = build_inspect_tab()
    preflight_tab = build_preflight_tab()
    study_tab = build_study_tab()
    results_tab = build_results_tab()

    demo = gr.TabbedInterface(
        [env_tab, inspect_tab, preflight_tab, study_tab, results_tab],
        ["Environment", "Model Inspection", "Preflight", "Study Runner", "Results"],
        title="TurboQuant Workflow Eval",
    )
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
