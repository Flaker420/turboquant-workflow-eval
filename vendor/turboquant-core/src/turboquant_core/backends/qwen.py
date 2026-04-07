"""
TurboQuant backend for Qwen3.5-9B hybrid architecture.

Compresses KV cache on the 8 GatedAttn layers (indices 3,7,11,15,19,23,27,31).
DeltaNet layers have no KV cache (opaque recurrent state in flash-linear-attention).

K → TQ_prod (MSE + QJL) for unbiased softmax(QK^T)
V → TQ_MSE only (weighted average, no inner product to debias)
"""

import math
from typing import Iterable, Sequence

from ..core import (
    CodebookRegistry, RotationCache, QJLProjection,
    tq_quantize_mse, tq_dequantize_mse, tq_quantize_prod,
)
import torch


class Qwen35KVBackend:
    """KV cache compression for Qwen3.5-9B GatedAttn layers."""

    MODEL_ID = "Qwen/Qwen3.5-9B"
    NUM_LAYERS = 32
    FULL_ATTN_INTERVAL = 4
    GA_KV_HEADS = 4
    GA_HEAD_DIM = 256

    def __init__(self, bit_width: int = 4, seed: int = 42,
                 device: torch.device = torch.device("cpu"), *,
                 num_layers: int = 32, full_attn_interval: int = 4,
                 kv_heads: int = 4, head_dim: int = 256,
                 key_strategy: str = "mse+qjl", value_strategy: str = "mse",
                 compressible_layers: Sequence[int] | None = None,
                 compressible_heads: Sequence[int] | None = None):
        _validate_strategies(key_strategy, value_strategy)
        self.num_layers = num_layers
        self.full_attn_interval = full_attn_interval
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.key_strategy = key_strategy
        self.value_strategy = value_strategy
        self.bit_width = bit_width

        default_ga = {i for i in range(num_layers)
                      if (i + 1) % full_attn_interval == 0}
        self.ga_indices = _resolve_compressible_layers(
            compressible_layers, default_ga, num_layers,
            backend_name="Qwen35KVBackend",
            require_subset_of_default=True,
        )
        self.compressible_layers = sorted(self.ga_indices)
        self.head_indices = _resolve_compressible_heads(
            compressible_heads, kv_heads, backend_name="Qwen35KVBackend",
        )
        self.compressible_heads = (
            sorted(self.head_indices) if self.head_indices is not None else None
        )
        d = head_dim

        if key_strategy == "mse+qjl":
            self.k_cb = CodebookRegistry.get(d, bit_width - 1, device)
            self.k_qjl = QJLProjection(d, seed=seed + 50, device=device)
        else:
            self.k_cb = CodebookRegistry.get(d, bit_width, device)
            self.k_qjl = None
        self.k_rot = RotationCache.get(d, seed, device)

        self.v_cb = CodebookRegistry.get(d, bit_width, device)
        self.v_rot = RotationCache.get(d, seed + 100, device)

    def is_compressible(self, layer_idx: int) -> bool:
        return layer_idx in self.ga_indices

    def compress(self, K: torch.Tensor, V: torch.Tensor, layer_idx: int) -> dict:
        """K, V shape: [batch, num_heads, seq_len, head_dim]"""
        assert self.is_compressible(layer_idx)
        return _compress_kv(self, K, V)

    def decompress_v(self, compressed: dict) -> torch.Tensor:
        return _decompress_v(self, compressed)

    def compute_attention_scores(self, Q: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased Q @ K^T from fresh Q and compressed K using QJL correction.

        Q shape: [batch, num_heads, q_len, head_dim]
        Returns: [batch, num_heads, q_len, kv_len]
        """
        return _compute_attention_scores(self, Q, compressed)


class Qwen3DenseKVBackend:
    """KV cache compression for Qwen3-8B (dense, all 36 layers uniform)."""

    MODEL_ID = "Qwen/Qwen3-8B"
    NUM_LAYERS = 36
    KV_HEADS = 8
    HEAD_DIM = 128

    def __init__(self, bit_width: int = 4, seed: int = 42,
                 device: torch.device = torch.device("cpu"), *,
                 num_layers: int = 36, kv_heads: int = 8, head_dim: int = 128,
                 key_strategy: str = "mse+qjl", value_strategy: str = "mse",
                 compressible_layers: Sequence[int] | None = None,
                 compressible_heads: Sequence[int] | None = None):
        _validate_strategies(key_strategy, value_strategy)
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.key_strategy = key_strategy
        self.value_strategy = value_strategy
        self.bit_width = bit_width

        self.ga_indices = _resolve_compressible_layers(
            compressible_layers, set(range(num_layers)), num_layers,
            backend_name="Qwen3DenseKVBackend",
            require_subset_of_default=False,
        )
        self.compressible_layers = sorted(self.ga_indices)
        self.head_indices = _resolve_compressible_heads(
            compressible_heads, kv_heads, backend_name="Qwen3DenseKVBackend",
        )
        self.compressible_heads = (
            sorted(self.head_indices) if self.head_indices is not None else None
        )

        d = head_dim
        if key_strategy == "mse+qjl":
            self.k_cb = CodebookRegistry.get(d, bit_width - 1, device)
            self.k_qjl = QJLProjection(d, seed=seed + 50, device=device)
        else:
            self.k_cb = CodebookRegistry.get(d, bit_width, device)
            self.k_qjl = None
        self.k_rot = RotationCache.get(d, seed, device)
        self.v_cb = CodebookRegistry.get(d, bit_width, device)
        self.v_rot = RotationCache.get(d, seed + 100, device)

    def is_compressible(self, layer_idx: int) -> bool:
        return layer_idx in self.ga_indices

    def compress(self, K, V, layer_idx):
        return _compress_kv(self, K, V)

    def decompress_v(self, compressed):
        return _decompress_v(self, compressed)

    def compute_attention_scores(self, Q: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased Q @ K^T from fresh Q and compressed K using QJL correction."""
        return _compute_attention_scores(self, Q, compressed)


class Qwen25DenseKVBackend:
    """KV cache compression for Qwen2.5-3B-Instruct (dense, all 36 layers uniform).

    Qwen2.5-3B uses GQA: 16 Q heads, 2 KV heads, head_dim=128.
    All 36 layers are standard attention with compressible KV cache.
    This is the recommended first ablation target due to small model size,
    full compressibility (f=1.0), and direct comparability with community
    benchmarks (tonbistudio, scos-lab, back2matching all tested on this model).
    """

    MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
    NUM_LAYERS = 36
    KV_HEADS = 2
    HEAD_DIM = 128

    def __init__(self, bit_width: int = 4, seed: int = 42,
                 device: torch.device = torch.device("cpu"), *,
                 num_layers: int = 36, kv_heads: int = 2, head_dim: int = 128,
                 key_strategy: str = "mse+qjl", value_strategy: str = "mse",
                 compressible_layers: Sequence[int] | None = None,
                 compressible_heads: Sequence[int] | None = None):
        _validate_strategies(key_strategy, value_strategy)
        self.num_layers = num_layers
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.key_strategy = key_strategy
        self.value_strategy = value_strategy
        self.bit_width = bit_width

        self.ga_indices = _resolve_compressible_layers(
            compressible_layers, set(range(num_layers)), num_layers,
            backend_name="Qwen25DenseKVBackend",
            require_subset_of_default=False,
        )
        self.compressible_layers = sorted(self.ga_indices)
        self.head_indices = _resolve_compressible_heads(
            compressible_heads, kv_heads, backend_name="Qwen25DenseKVBackend",
        )
        self.compressible_heads = (
            sorted(self.head_indices) if self.head_indices is not None else None
        )

        d = head_dim
        if key_strategy == "mse+qjl":
            self.k_cb = CodebookRegistry.get(d, bit_width - 1, device)
            self.k_qjl = QJLProjection(d, seed=seed + 50, device=device)
        else:
            self.k_cb = CodebookRegistry.get(d, bit_width, device)
            self.k_qjl = None
        self.k_rot = RotationCache.get(d, seed, device)
        self.v_cb = CodebookRegistry.get(d, bit_width, device)
        self.v_rot = RotationCache.get(d, seed + 100, device)

    def is_compressible(self, layer_idx: int) -> bool:
        return layer_idx in self.ga_indices

    def compress(self, K, V, layer_idx):
        return _compress_kv(self, K, V)

    def decompress_v(self, compressed):
        return _decompress_v(self, compressed)

    def compute_attention_scores(self, Q: torch.Tensor, compressed: dict) -> torch.Tensor:
        return _compute_attention_scores(self, Q, compressed)


_VALID_KEY_STRATEGIES = {"mse", "mse+qjl"}
_VALID_VALUE_STRATEGIES = {"mse"}


def _validate_strategies(key_strategy: str, value_strategy: str):
    if key_strategy not in _VALID_KEY_STRATEGIES:
        raise ValueError(
            f"Invalid key_strategy {key_strategy!r}. "
            f"Must be one of {_VALID_KEY_STRATEGIES}"
        )
    if value_strategy not in _VALID_VALUE_STRATEGIES:
        raise ValueError(
            f"Invalid value_strategy {value_strategy!r}. "
            f"Must be one of {_VALID_VALUE_STRATEGIES}"
        )


def _resolve_compressible_layers(
    user_value: Sequence[int] | None,
    default_indices: Iterable[int],
    num_layers: int,
    *,
    backend_name: str,
    require_subset_of_default: bool,
) -> set[int]:
    """Validate and normalize a user-supplied ``compressible_layers`` list.

    When ``user_value`` is None, returns the default index set unchanged.
    Otherwise, validates each entry against the per-backend constraints and
    returns the normalized set. Raises ValueError on any invalid input.
    """
    default_set = set(default_indices)
    if user_value is None:
        return default_set

    seen: set[int] = set()
    bad_type: list = []
    bad_range: list[int] = []
    bad_subset: list[int] = []
    for idx in user_value:
        if isinstance(idx, bool) or not isinstance(idx, int):
            bad_type.append(idx)
            continue
        if idx < 0 or idx >= num_layers:
            bad_range.append(idx)
            continue
        if require_subset_of_default and idx not in default_set:
            bad_subset.append(idx)
            continue
        seen.add(idx)

    if bad_type:
        raise ValueError(
            f"{backend_name}: compressible_layers must contain ints; "
            f"got non-int entries {bad_type!r}."
        )
    if bad_range:
        raise ValueError(
            f"{backend_name}: compressible_layers indices {sorted(set(bad_range))} "
            f"are out of range [0, {num_layers})."
        )
    if bad_subset:
        raise ValueError(
            f"{backend_name}: compressible_layers indices {sorted(set(bad_subset))} "
            f"are not GatedAttn layers (DeltaNet has no KV cache to compress). "
            f"Valid GatedAttn indices for this backend: {sorted(default_set)}."
        )
    if not seen:
        raise ValueError(
            f"{backend_name}: compressible_layers must be non-empty when provided."
        )
    return seen


def _resolve_compressible_heads(
    user_value: Sequence[int] | None,
    num_kv_heads: int,
    *,
    backend_name: str,
) -> frozenset[int] | None:
    """Validate and normalize a user-supplied ``compressible_heads`` list.

    ``None`` means "compress every head" and short-circuits any per-head
    slicing in ``compress`` / ``decompress_v`` / ``compute_attention_scores``.
    Any non-None value must be a non-empty, duplicate-free list of ints in
    ``[0, num_kv_heads)``. Returns a frozenset for fast membership tests.
    """
    if user_value is None:
        return None

    seen: set[int] = set()
    duplicates: list[int] = []
    bad_type: list = []
    bad_range: list[int] = []
    for idx in user_value:
        if isinstance(idx, bool) or not isinstance(idx, int):
            bad_type.append(idx)
            continue
        if idx < 0 or idx >= num_kv_heads:
            bad_range.append(idx)
            continue
        if idx in seen:
            duplicates.append(idx)
            continue
        seen.add(idx)

    if bad_type:
        raise ValueError(
            f"{backend_name}: compressible_heads must contain ints; "
            f"got non-int entries {bad_type!r}."
        )
    if bad_range:
        raise ValueError(
            f"{backend_name}: compressible_heads indices {sorted(set(bad_range))} "
            f"are out of range [0, {num_kv_heads})."
        )
    if duplicates:
        raise ValueError(
            f"{backend_name}: compressible_heads contains duplicate indices "
            f"{sorted(set(duplicates))}."
        )
    if not seen:
        raise ValueError(
            f"{backend_name}: compressible_heads must be non-empty when provided."
        )
    return frozenset(seen)


# ---------------------------------------------------------------------------
# Shared compress / decompress / score helpers
# ---------------------------------------------------------------------------
#
# All three Qwen backends share the same K/V shape convention
# ``[batch, num_heads, seq_len, head_dim]`` and the same set of quantization
# primitives (``k_cb``, ``v_cb``, ``k_rot``, ``v_rot``, optional ``k_qjl``).
# The helpers below operate on any backend instance that exposes these
# attributes and ``key_strategy``. When ``backend.head_indices`` is ``None``
# the returned dicts are bit-identical to the pre-head-masking schema, so
# downstream callers that don't set ``compressible_heads`` see no change.


def _compress_all_heads(backend, K: "torch.Tensor", V: "torch.Tensor") -> dict:
    """Compress every head in ``K``/``V`` (legacy, pre-head-masking path)."""
    b, nh, sl, hd = K.shape
    Kf, Vf = K.reshape(-1, hd), V.reshape(-1, hd)
    if backend.key_strategy == "mse+qjl":
        k_mse, k_qjl, k_rn, k_n = tq_quantize_prod(Kf, backend.k_cb, backend.k_rot, backend.k_qjl)
        v_idx, v_n = tq_quantize_mse(Vf, backend.v_cb, backend.v_rot)
        return {"k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rn, "k_n": k_n,
                "v_idx": v_idx, "v_n": v_n, "shape": K.shape}
    k_idx, k_n = tq_quantize_mse(Kf, backend.k_cb, backend.k_rot)
    v_idx, v_n = tq_quantize_mse(Vf, backend.v_cb, backend.v_rot)
    return {"k_mse": k_idx, "k_n": k_n,
            "v_idx": v_idx, "v_n": v_n, "shape": K.shape}


def _compress_kv(backend, K: "torch.Tensor", V: "torch.Tensor") -> dict:
    """Compress ``K``/``V``. If ``backend.head_indices`` is set, only those
    KV heads are quantized; the remaining heads are stored in full precision
    and carried through ``compressed["k_raw"]`` / ``compressed["v_raw"]``.
    """
    if backend.head_indices is None:
        return _compress_all_heads(backend, K, V)

    b, nh, sl, hd = K.shape
    mask = tuple(sorted(backend.head_indices))
    complement = tuple(h for h in range(nh) if h not in backend.head_indices)

    # Advanced indexing along dim=1 gives [b, len(mask), sl, hd] contiguous copies.
    mask_idx = torch.tensor(mask, device=K.device, dtype=torch.long)
    compl_idx = torch.tensor(complement, device=K.device, dtype=torch.long) if complement else None

    K_m = K.index_select(1, mask_idx)
    V_m = V.index_select(1, mask_idx)
    if compl_idx is not None:
        K_c = K.index_select(1, compl_idx).contiguous()
        V_c = V.index_select(1, compl_idx).contiguous()
    else:
        K_c = K.new_zeros(b, 0, sl, hd)
        V_c = V.new_zeros(b, 0, sl, hd)

    # Reuse the legacy path on the masked-head slice.
    sub = _compress_all_heads(backend, K_m.contiguous(), V_m.contiguous())
    sub["head_mask"] = mask
    sub["head_complement"] = complement
    sub["k_raw"] = K_c
    sub["v_raw"] = V_c
    sub["full_shape"] = K.shape  # [b, nh, sl, hd] including complement heads
    return sub


def _decompress_v(backend, compressed: dict) -> "torch.Tensor":
    """Reassemble the full-head V tensor from a compressed dict."""
    Vf = tq_dequantize_mse(
        compressed["v_idx"], compressed["v_n"], backend.v_cb, backend.v_rot
    )
    if "head_mask" not in compressed:
        return Vf.reshape(compressed["shape"])

    b, nh, sl, hd = compressed["full_shape"]
    mask = list(compressed["head_mask"])
    complement = list(compressed["head_complement"])
    V_m = Vf.reshape(b, len(mask), sl, hd)
    V_full = V_m.new_empty(b, nh, sl, hd)
    if mask:
        V_full[:, mask, :, :] = V_m
    if complement:
        V_full[:, complement, :, :] = compressed["v_raw"]
    return V_full


def _compute_attention_scores(backend, Q: "torch.Tensor", compressed: dict) -> "torch.Tensor":
    """Compute ``Q @ K^T`` using MSE(+QJL) for compressed heads and the raw
    ``K`` for complement heads when a head mask is present. Returns a tensor
    of shape ``[b, nh, q_len, kv_len]`` matching the full pre-masking layout.
    """
    if "head_mask" not in compressed:
        b, nh, kv_len, hd = compressed["shape"]
        q_len = Q.shape[2]
        K_mse = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"], backend.k_cb, backend.k_rot
        ).reshape(compressed["shape"])
        scores = Q @ K_mse.transpose(-2, -1)
        if backend.key_strategy != "mse+qjl":
            return scores
        Q_flat = Q.reshape(b * nh * q_len, hd)
        Q_qjl = backend.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
        k_qjl = compressed["k_qjl"].reshape(b * nh, kv_len, hd)
        correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
        correction = correction * (math.pi / (2 * hd))
        correction = correction * compressed["k_rn"].reshape(b * nh, 1, kv_len)
        correction = correction * compressed["k_n"].reshape(b * nh, 1, kv_len)
        correction = correction.reshape(b, nh, q_len, kv_len)
        return scores + correction

    # Per-head mask path: compute masked-head scores via the compressed
    # pipeline and complement-head scores via direct Q @ K_raw^T, then
    # scatter the sub-results back into a full [b, nh, q_len, kv_len] tensor.
    b, nh, sl_full, hd = compressed["full_shape"]
    q_len = Q.shape[2]
    mask = list(compressed["head_mask"])
    complement = list(compressed["head_complement"])
    m = len(mask)

    # Masked heads: run the existing MSE(+QJL) score computation on the
    # sub-tensor by building a temporary "compressed" dict that advertises
    # shape = [b, m, sl, hd] and no head mask.
    sub = {k: compressed[k] for k in compressed
           if k not in ("head_mask", "head_complement", "k_raw", "v_raw", "full_shape")}
    sub["shape"] = (b, m, sl_full, hd)
    Q_m = Q.index_select(1, torch.tensor(mask, device=Q.device, dtype=torch.long)).contiguous()
    scores_masked = _compute_attention_scores(backend, Q_m, sub)  # [b, m, q_len, sl]

    scores_full = scores_masked.new_empty(b, nh, q_len, sl_full)
    scores_full[:, mask, :, :] = scores_masked
    if complement:
        compl_idx = torch.tensor(complement, device=Q.device, dtype=torch.long)
        Q_c = Q.index_select(1, compl_idx).contiguous()
        K_c = compressed["k_raw"]  # [b, len(complement), sl_full, hd]
        scores_compl = Q_c @ K_c.transpose(-2, -1)
        scores_full[:, complement, :, :] = scores_compl
    return scores_full
