"""CLI entry point: ``python -m turboquant_workflow_eval``.

The CLI loads a Python study module, applies any per-knob or per-policy
overrides via :func:`turboquant_workflow_eval.schema.replace_path` (which
recursively rebuilds frozen dataclasses), then dispatches to the runner,
the dry-run validator, or the rescoring path.

Every knob in :class:`StudyConfig` has a dedicated ``--KNOB`` flag with an
argparse type and a help string; ``--set`` and ``--set-policy`` remain as
escape hatches for the long tail of paths.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from typing import Any

from .schema import (
    PolicyConfig,
    PolicySettings,
    StudyConfig,
    replace_path,
)


# ---------------------------------------------------------------------------
# argparse value parsers
# ---------------------------------------------------------------------------


def parse_int_list(text: str) -> tuple[int, ...]:
    """Parse a comma-separated int list (e.g. ``3,7,11,15``)."""
    items = [s.strip() for s in text.split(",") if s.strip()]
    try:
        return tuple(int(s) for s in items)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"comma-separated int list expected, got {text!r}"
        ) from exc


def _parse_str_list(text: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in text.split(",") if s.strip())


def _coerce_str_value(raw: str) -> Any:
    """Best-effort literal coercion mirroring the legacy ``--set`` semantics."""
    s = raw.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


# ---------------------------------------------------------------------------
# argparse Actions for KEY=VALUE pairs
# ---------------------------------------------------------------------------


class KeyValuePolicyAppend(argparse.Action):
    """Append ``NAME=VALUE`` pairs onto a list of (name, raw_value) tuples.

    Used by every per-policy ``--KNOB-for`` flag. The CLI handler later
    coerces ``raw_value`` to the right type per knob.
    """

    def __call__(self, parser, namespace, values, option_string=None):  # type: ignore[override]
        if "=" not in values:
            raise argparse.ArgumentError(
                self, f"expected NAME=VALUE, got {values!r}"
            )
        name, _, raw = values.partition("=")
        name = name.strip()
        if not name:
            raise argparse.ArgumentError(self, f"empty policy name in {values!r}")
        existing = getattr(namespace, self.dest, None) or []
        existing.append((name, raw.strip()))
        setattr(namespace, self.dest, existing)


# ---------------------------------------------------------------------------
# Per-policy override application
# ---------------------------------------------------------------------------


# (CLI flag knob name, PolicySettings field name, value coercer)
_PER_POLICY_KNOBS: list[tuple[str, str, Any]] = [
    ("bit-width", "bit_width", int),
    ("seed", "seed", int),
    ("residual-window", "residual_window", int),
    ("key-strategy", "key_strategy", str),
    ("value-strategy", "value_strategy", str),
    ("compressible-layers", "compressible_layers", parse_int_list),
    ("profile", "profile", str),
]


def _apply_global_settings_overrides(
    study: StudyConfig, updates: dict[str, Any]
) -> StudyConfig:
    """Apply a dict of ``PolicySettings`` field overrides to every policy."""
    if not updates:
        return study
    new_policies = []
    for p in study.policies:
        new_settings = replace(p.settings, **updates)
        new_policies.append(replace(p, settings=new_settings))
    return replace(study, policies=tuple(new_policies))


def _apply_per_policy_overrides(
    study: StudyConfig,
    field_name: str,
    pairs: list[tuple[str, Any]],
) -> StudyConfig:
    """Apply ``--KNOB-for NAME=VALUE`` overrides for one PolicySettings field."""
    if not pairs:
        return study
    by_name: dict[str, list[Any]] = {}
    for name, value in pairs:
        by_name.setdefault(name, []).append(value)

    known = {p.name for p in study.policies}
    unknown = set(by_name) - known
    if unknown:
        raise SystemExit(
            f"--{field_name.replace('_', '-')}-for: unknown policy name(s) "
            f"{sorted(unknown)}. Known policies: {sorted(known)}"
        )

    new_policies = []
    for p in study.policies:
        if p.name in by_name:
            # Last write wins if a flag is repeated for the same policy.
            value = by_name[p.name][-1]
            new_settings = replace(p.settings, **{field_name: value})
            new_policies.append(replace(p, settings=new_settings))
        else:
            new_policies.append(p)
    return replace(study, policies=tuple(new_policies))


def apply_set_overrides(study: StudyConfig, items: list[str]) -> StudyConfig:
    """Apply ``--set DOT.PATH=VALUE`` overrides via ``replace_path``."""
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--set: expected KEY=VALUE, got {item!r}")
        key, raw = item.split("=", 1)
        study = replace_path(study, key.strip(), _coerce_str_value(raw))
    return study


def apply_set_policy_overrides(
    study: StudyConfig, items: list[str]
) -> StudyConfig:
    """Apply ``--set-policy NAME.DOT.PATH=VALUE`` overrides.

    The first segment is a policy name (or ``*`` for every policy); the
    remaining segments form a dot-path inside :class:`PolicyConfig`.
    """
    for item in items:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise SystemExit(
                f"--set-policy: expected '<name|*>.<dot.key>=value', got {item!r}"
            )
        key, raw = item.split("=", 1)
        target_name, _, dot_key = key.strip().partition(".")
        if not dot_key:
            raise SystemExit(f"--set-policy: missing nested key in {item!r}")
        value = _coerce_str_value(raw)
        new_policies = []
        matched = False
        for p in study.policies:
            if target_name in ("*", p.name):
                p = replace_path(p, dot_key, value)
                matched = True
            new_policies.append(p)
        if not matched and target_name != "*":
            raise SystemExit(
                f"--set-policy: no policy matches {target_name!r}. "
                f"Known policies: {[p.name for p in study.policies]}"
            )
        study = replace(study, policies=tuple(new_policies))
    return study


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="turboquant_workflow_eval",
        description="Run a workflow evaluation study comparing compression policies.",
    )

    # --- Study selection / output ---
    parser.add_argument(
        "--study",
        default=None,
        help=(
            "Path to a Python study module (e.g. configs/studies/default_qwen35_9b.py) "
            "or a 'package.module:STUDY' import path. Required unless --rescore."
        ),
    )
    parser.add_argument("--output-dir", default="outputs",
                        help="Directory for result artefacts (default: outputs/)")
    parser.add_argument(
        "--model",
        default=None,
        help="Override the study's model with a different model module path.",
    )
    parser.add_argument(
        "--policies",
        default=None,
        help="Comma-separated list of policy module paths that overrides study.policies.",
    )

    # --- Execution modes ---
    parser.add_argument("--single", action="store_true",
                        help="Quick smoke test: first matching prompt, first enabled policy, 1 repetition")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate all configs and print execution plan without touching the GPU")
    parser.add_argument("--rescore", default=None, metavar="ROWS_JSONL",
                        help="Re-score existing results with new thresholds (no GPU). Use --set for threshold overrides.")

    # --- Prompt filters ---
    parser.add_argument("--prompt-id", action="append", dest="prompt_ids", default=None,
                        help="Run only the specified prompt ID (repeatable)")
    parser.add_argument("--prompt-category", action="append", dest="prompt_categories", default=None,
                        help="Run only prompts in the specified category (repeatable)")
    parser.add_argument("--prompt-filter", default=None, metavar="REGEX",
                        help="Filter prompts by regex on id/title")

    # --- Global PolicySettings knobs (apply to every policy) ---
    g = parser.add_argument_group("global policy settings (apply to every policy)")
    g.add_argument("--bit-width", type=int, default=None,
                   help="Override PolicySettings.bit_width on every policy.")
    g.add_argument("--seed", type=int, default=None,
                   help="Override PolicySettings.seed on every policy.")
    g.add_argument("--residual-window", type=int, default=None,
                   help="Override PolicySettings.residual_window on every policy.")
    g.add_argument("--key-strategy", choices=("mse", "mse+qjl"), default=None,
                   help="Override PolicySettings.key_strategy on every policy.")
    g.add_argument("--value-strategy", choices=("mse",), default=None,
                   help="Override PolicySettings.value_strategy on every policy.")
    g.add_argument("--compressible-layers", type=parse_int_list, default=None,
                   metavar="3,7,11,...",
                   help="Override PolicySettings.compressible_layers on every policy "
                        "(comma-separated integer indices).")
    g.add_argument("--profile", default=None,
                   help="Override PolicySettings.profile on every policy.")

    # --- Per-policy knobs ---
    pp = parser.add_argument_group("per-policy overrides (NAME=VALUE, repeatable)")
    for cli, _field, _coerce in _PER_POLICY_KNOBS:
        pp.add_argument(
            f"--{cli}-for",
            action=KeyValuePolicyAppend,
            dest=f"{cli.replace('-', '_')}_for",
            default=[],
            metavar="NAME=VALUE",
            help=f"Override PolicySettings.{_field} on a single policy by name.",
        )

    # --- Runtime knobs ---
    r = parser.add_argument_group("runtime")
    r.add_argument("--max-input-tokens", type=int, default=None)
    r.add_argument("--max-new-tokens", type=int, default=None)
    r.add_argument("--temperature", type=float, default=None)
    r.add_argument("--top-p", type=float, default=None)
    r.add_argument("--repetitions", type=int, default=None)
    r.add_argument("--no-cache", action="store_true",
                   help="Set runtime.use_cache=False.")
    r.add_argument("--shuffle-policies", action="store_true",
                   help="Set runtime.shuffle_policies=True.")
    r.add_argument("--shuffle-seed", type=int, default=None)
    r.add_argument("--baseline-policy", default=None,
                   help="Override study.baseline_policy_name.")

    # --- Threshold knobs ---
    t = parser.add_argument_group("thresholds")
    t.add_argument("--latency-yellow-pct", type=float, default=None)
    t.add_argument("--latency-red-pct", type=float, default=None)
    t.add_argument("--similarity-yellow", type=float, default=None)
    t.add_argument("--similarity-red", type=float, default=None)
    t.add_argument("--output-length-yellow-pct", type=float, default=None)
    t.add_argument("--output-length-red-pct", type=float, default=None)

    # --- Early-stop knobs ---
    e = parser.add_argument_group("early stop")
    e.add_argument("--max-red-verdicts", type=int, default=None)
    e.add_argument("--max-error-rate", type=float, default=None)

    # --- Escape hatches ---
    parser.add_argument(
        "--set",
        action="append",
        dest="overrides",
        metavar="DOT.PATH=VALUE",
        default=[],
        help=(
            "Escape hatch: override any StudyConfig field via dot-path "
            "(e.g. --set runtime.max_new_tokens=128). Repeatable. "
            "Applied via dataclasses.replace recursively."
        ),
    )
    parser.add_argument(
        "--set-policy",
        action="append",
        dest="policy_overrides",
        metavar="NAME.DOT.KEY=VALUE",
        default=[],
        help=(
            "Escape hatch: per-policy dot-path override. "
            "First segment is a policy name (or '*' for every policy). "
            "Examples: --set-policy turboquant_safe.settings.bit_width=8, "
            "--set-policy '*.settings.key_strategy=mse', "
            "--set-policy baseline.enabled=false. Repeatable."
        ),
    )

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _apply_overrides(study: StudyConfig, args: argparse.Namespace) -> StudyConfig:
    """Apply every CLI override to the loaded study, in deterministic order."""
    # 1. --model: replace study.model wholesale.
    if args.model:
        from .loader import load_model_module
        study = replace(study, model=load_model_module(args.model))

    # 2. --policies: replace study.policies wholesale.
    if args.policies:
        from .loader import load_policy_module
        paths = [p.strip() for p in args.policies.split(",") if p.strip()]
        new_policies = tuple(load_policy_module(p) for p in paths)
        study = replace(study, policies=new_policies)

    # 3. Global PolicySettings overrides.
    settings_updates: dict[str, Any] = {}
    if args.bit_width is not None:
        settings_updates["bit_width"] = args.bit_width
    if args.seed is not None:
        settings_updates["seed"] = args.seed
    if args.residual_window is not None:
        settings_updates["residual_window"] = args.residual_window
    if args.key_strategy is not None:
        settings_updates["key_strategy"] = args.key_strategy
    if args.value_strategy is not None:
        settings_updates["value_strategy"] = args.value_strategy
    if args.compressible_layers is not None:
        settings_updates["compressible_layers"] = args.compressible_layers
    if args.profile is not None:
        settings_updates["profile"] = args.profile
    study = _apply_global_settings_overrides(study, settings_updates)

    # 4. Per-policy --KNOB-for overrides.
    for cli, field_name, coerce in _PER_POLICY_KNOBS:
        attr = f"{cli.replace('-', '_')}_for"
        raw_pairs: list[tuple[str, str]] = getattr(args, attr) or []
        if not raw_pairs:
            continue
        coerced = [(name, coerce(value)) for name, value in raw_pairs]
        study = _apply_per_policy_overrides(study, field_name, coerced)

    # 5. Runtime overrides.
    runtime_updates: dict[str, Any] = {}
    if args.max_input_tokens is not None:
        runtime_updates["max_input_tokens"] = args.max_input_tokens
    if args.max_new_tokens is not None:
        runtime_updates["max_new_tokens"] = args.max_new_tokens
    if args.temperature is not None:
        runtime_updates["temperature"] = args.temperature
    if args.top_p is not None:
        runtime_updates["top_p"] = args.top_p
    if args.repetitions is not None:
        runtime_updates["repetitions"] = args.repetitions
    if args.no_cache:
        runtime_updates["use_cache"] = False
    if args.shuffle_policies:
        runtime_updates["shuffle_policies"] = True
    if args.shuffle_seed is not None:
        runtime_updates["shuffle_seed"] = args.shuffle_seed
    if runtime_updates:
        new_runtime = replace(study.runtime, **runtime_updates)
        study = replace(study, runtime=new_runtime)

    # 6. Threshold overrides.
    threshold_updates: dict[str, Any] = {}
    for f in (
        "latency_yellow_pct", "latency_red_pct",
        "similarity_yellow", "similarity_red",
        "output_length_yellow_pct", "output_length_red_pct",
    ):
        v = getattr(args, f, None)
        if v is not None:
            threshold_updates[f] = v
    if threshold_updates:
        new_thresholds = replace(study.thresholds, **threshold_updates)
        study = replace(study, thresholds=new_thresholds)

    # 7. Early-stop overrides.
    early_updates: dict[str, Any] = {}
    if args.max_red_verdicts is not None:
        early_updates["max_red_verdicts"] = args.max_red_verdicts
    if args.max_error_rate is not None:
        early_updates["max_error_rate"] = args.max_error_rate
    if early_updates:
        new_early = replace(study.early_stop, **early_updates)
        study = replace(study, early_stop=new_early)

    # 8. baseline-policy override.
    if args.baseline_policy is not None:
        study = replace(study, baseline_policy_name=args.baseline_policy)

    # 9. --single forces repetitions=1.
    if args.single:
        new_runtime = replace(study.runtime, repetitions=1)
        study = replace(study, runtime=new_runtime)

    # 10. Generic escape hatches (last so they win over typed flags).
    if args.overrides:
        study = apply_set_overrides(study, args.overrides)
    if args.policy_overrides:
        study = apply_set_policy_overrides(study, args.policy_overrides)

    return study


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # --- Re-score mode ---
    if args.rescore:
        from .rescoring import rescore

        rescore(
            rows_jsonl_path=args.rescore,
            study_config=args.study,
            overrides=args.overrides or None,
            output_dir=args.output_dir if args.output_dir != "outputs" else None,
        )
        return

    if not args.study:
        parser.error("--study is required unless --rescore is given")

    # --- Load study and apply overrides ---
    from .loader import load_study_module

    study = load_study_module(args.study)
    study = _apply_overrides(study, args)

    # --- Dry-run mode ---
    if args.dry_run:
        from .validation import dry_run

        exit_code = dry_run(
            study,
            prompt_ids=args.prompt_ids,
            prompt_categories=args.prompt_categories,
            prompt_pattern=args.prompt_filter,
        )
        sys.exit(exit_code)

    # --- Normal run ---
    from .study import run_workflow_study

    summary = run_workflow_study(
        study,
        output_dir=args.output_dir,
        prompt_ids=args.prompt_ids,
        prompt_categories=args.prompt_categories,
        prompt_pattern=args.prompt_filter,
        single_mode=args.single,
        study_config_path=args.study,
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
