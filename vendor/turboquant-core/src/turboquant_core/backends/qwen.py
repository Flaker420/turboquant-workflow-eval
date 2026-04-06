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
                 compressible_layers: Sequence[int] | None = None):
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
        b, nh, sl, hd = K.shape
        Kf, Vf = K.reshape(-1, hd), V.reshape(-1, hd)
        if self.key_strategy == "mse+qjl":
            k_mse, k_qjl, k_rn, k_n = tq_quantize_prod(Kf, self.k_cb, self.k_rot, self.k_qjl)
            v_idx, v_n = tq_quantize_mse(Vf, self.v_cb, self.v_rot)
            return {"k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rn, "k_n": k_n,
                    "v_idx": v_idx, "v_n": v_n, "shape": K.shape}
        else:
            k_idx, k_n = tq_quantize_mse(Kf, self.k_cb, self.k_rot)
            v_idx, v_n = tq_quantize_mse(Vf, self.v_cb, self.v_rot)
            return {"k_mse": k_idx, "k_n": k_n,
                    "v_idx": v_idx, "v_n": v_n, "shape": K.shape}

    def decompress_v(self, compressed: dict) -> torch.Tensor:
        Vf = tq_dequantize_mse(compressed["v_idx"], compressed["v_n"], self.v_cb, self.v_rot)
        return Vf.reshape(compressed["shape"])

    def compute_attention_scores(self, Q: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased Q @ K^T from fresh Q and compressed K using QJL correction.

        Q shape: [batch, num_heads, q_len, head_dim]
        Returns: [batch, num_heads, q_len, kv_len]
        """
        b, nh, kv_len, hd = compressed["shape"]
        q_len = Q.shape[2]

        K_mse = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"], self.k_cb, self.k_rot
        ).reshape(compressed["shape"])
        scores_mse = Q @ K_mse.transpose(-2, -1)

        if self.key_strategy == "mse+qjl":
            Q_flat = Q.reshape(b * nh * q_len, hd)
            Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
            k_qjl = compressed["k_qjl"].reshape(b * nh, kv_len, hd)

            correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
            correction = correction * (math.pi / (2 * hd))
            correction = correction * compressed["k_rn"].reshape(b * nh, 1, kv_len)
            correction = correction * compressed["k_n"].reshape(b * nh, 1, kv_len)
            correction = correction.reshape(b, nh, q_len, kv_len)

            return scores_mse + correction
        else:
            return scores_mse


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
                 compressible_layers: Sequence[int] | None = None):
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
        b, nh, sl, hd = K.shape
        Kf, Vf = K.reshape(-1, hd), V.reshape(-1, hd)
        if self.key_strategy == "mse+qjl":
            k_mse, k_qjl, k_rn, k_n = tq_quantize_prod(Kf, self.k_cb, self.k_rot, self.k_qjl)
            v_idx, v_n = tq_quantize_mse(Vf, self.v_cb, self.v_rot)
            return {"k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rn, "k_n": k_n,
                    "v_idx": v_idx, "v_n": v_n, "shape": K.shape}
        else:
            k_idx, k_n = tq_quantize_mse(Kf, self.k_cb, self.k_rot)
            v_idx, v_n = tq_quantize_mse(Vf, self.v_cb, self.v_rot)
            return {"k_mse": k_idx, "k_n": k_n,
                    "v_idx": v_idx, "v_n": v_n, "shape": K.shape}

    def decompress_v(self, compressed):
        Vf = tq_dequantize_mse(compressed["v_idx"], compressed["v_n"], self.v_cb, self.v_rot)
        return Vf.reshape(compressed["shape"])

    def compute_attention_scores(self, Q: torch.Tensor, compressed: dict) -> torch.Tensor:
        """Compute unbiased Q @ K^T from fresh Q and compressed K using QJL correction."""
        b, nh, kv_len, hd = compressed["shape"]
        q_len = Q.shape[2]

        K_mse = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"], self.k_cb, self.k_rot
        ).reshape(compressed["shape"])
        scores_mse = Q @ K_mse.transpose(-2, -1)

        if self.key_strategy == "mse+qjl":
            Q_flat = Q.reshape(b * nh * q_len, hd)
            Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
            k_qjl = compressed["k_qjl"].reshape(b * nh, kv_len, hd)

            correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
            correction = correction * (math.pi / (2 * hd))
            correction = correction * compressed["k_rn"].reshape(b * nh, 1, kv_len)
            correction = correction * compressed["k_n"].reshape(b * nh, 1, kv_len)
            correction = correction.reshape(b, nh, q_len, kv_len)

            return scores_mse + correction
        else:
            return scores_mse


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
                 compressible_layers: Sequence[int] | None = None):
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
        b, nh, sl, hd = K.shape
        Kf, Vf = K.reshape(-1, hd), V.reshape(-1, hd)
        if self.key_strategy == "mse+qjl":
            k_mse, k_qjl, k_rn, k_n = tq_quantize_prod(Kf, self.k_cb, self.k_rot, self.k_qjl)
            v_idx, v_n = tq_quantize_mse(Vf, self.v_cb, self.v_rot)
            return {"k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rn, "k_n": k_n,
                    "v_idx": v_idx, "v_n": v_n, "shape": K.shape}
        else:
            k_idx, k_n = tq_quantize_mse(Kf, self.k_cb, self.k_rot)
            v_idx, v_n = tq_quantize_mse(Vf, self.v_cb, self.v_rot)
            return {"k_mse": k_idx, "k_n": k_n,
                    "v_idx": v_idx, "v_n": v_n, "shape": K.shape}

    def decompress_v(self, compressed):
        Vf = tq_dequantize_mse(compressed["v_idx"], compressed["v_n"], self.v_cb, self.v_rot)
        return Vf.reshape(compressed["shape"])

    def compute_attention_scores(self, Q: torch.Tensor, compressed: dict) -> torch.Tensor:
        b, nh, kv_len, hd = compressed["shape"]
        q_len = Q.shape[2]

        K_mse = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"], self.k_cb, self.k_rot
        ).reshape(compressed["shape"])
        scores_mse = Q @ K_mse.transpose(-2, -1)

        if self.key_strategy == "mse+qjl":
            Q_flat = Q.reshape(b * nh * q_len, hd)
            Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
            k_qjl = compressed["k_qjl"].reshape(b * nh, kv_len, hd)

            correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
            correction = correction * (math.pi / (2 * hd))
            correction = correction * compressed["k_rn"].reshape(b * nh, 1, kv_len)
            correction = correction * compressed["k_n"].reshape(b * nh, 1, kv_len)
            correction = correction.reshape(b, nh, q_len, kv_len)

            return scores_mse + correction
        else:
            return scores_mse


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
