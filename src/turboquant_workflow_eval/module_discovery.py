from __future__ import annotations

from typing import Any

from .constants import (
    ATTENTION_EXCLUDE_HINTS,
    ATTENTION_NAME_HINTS,
    K_PROJ_HINTS,
    PACKED_QKV_HINTS,
    Q_PROJ_HINTS,
    V_PROJ_HINTS,
)
from .types import AttentionBlockRef


def _norm(text: str) -> str:
    return str(text).lower().replace("-", "_")


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    text = _norm(text)
    return any(needle in text for needle in needles)


def _looks_attention_like(module_path: str, module: Any) -> bool:
    joined = f"{module_path} {module.__class__.__name__}"
    if _contains_any(joined, ATTENTION_EXCLUDE_HINTS):
        return False
    return _contains_any(joined, ATTENTION_NAME_HINTS)


def _projection_child_map(module: Any) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for child_name, _child in module.named_children():
        name = _norm(child_name)
        if _contains_any(name, PACKED_QKV_HINTS):
            mapping["packed_qkv"] = child_name
        elif _contains_any(name, Q_PROJ_HINTS):
            mapping.setdefault("q", child_name)
        elif _contains_any(name, K_PROJ_HINTS):
            mapping.setdefault("k", child_name)
        elif _contains_any(name, V_PROJ_HINTS):
            mapping.setdefault("v", child_name)
    return mapping


def discover_attention_blocks(model_root: Any, expected_count: int | None = None) -> list[AttentionBlockRef]:
    candidates = []
    for module_path, module in model_root.named_modules():
        if not module_path:
            continue
        child_map = _projection_child_map(module)
        has_qkv = {"q", "k", "v"}.issubset(child_map) or ("packed_qkv" in child_map)
        if has_qkv and _looks_attention_like(module_path, module):
            candidates.append((module_path, module, child_map))

    candidates.sort(key=lambda item: item[0].count("."))
    kept = []
    for module_path, module, child_map in candidates:
        if any(module_path == kept_path or module_path.startswith(kept_path + ".") for kept_path, _m, _c in kept):
            continue
        kept.append((module_path, module, child_map))

    results: list[AttentionBlockRef] = []
    for index, (module_path, module, child_map) in enumerate(kept):
        notes: list[str] = []
        if "packed_qkv" in child_map and not {"q", "k", "v"}.issubset(child_map):
            notes.append("packed_qkv_detected")
        results.append(
            AttentionBlockRef(
                index=index,
                module_path=module_path,
                class_name=module.__class__.__name__,
                q_proj_path=f"{module_path}.{child_map['q']}" if "q" in child_map else None,
                k_proj_path=f"{module_path}.{child_map['k']}" if "k" in child_map else None,
                v_proj_path=f"{module_path}.{child_map['v']}" if "v" in child_map else None,
                packed_qkv_path=f"{module_path}.{child_map['packed_qkv']}" if "packed_qkv" in child_map else None,
                notes=tuple(notes),
            )
        )

    if expected_count is not None and len(results) != expected_count:
        warning = f"expected_{expected_count}_found_{len(results)}"
        results = [
            AttentionBlockRef(
                index=ref.index,
                module_path=ref.module_path,
                class_name=ref.class_name,
                q_proj_path=ref.q_proj_path,
                k_proj_path=ref.k_proj_path,
                v_proj_path=ref.v_proj_path,
                packed_qkv_path=ref.packed_qkv_path,
                notes=tuple(list(ref.notes) + [warning]),
            )
            for ref in results
        ]
    return results
