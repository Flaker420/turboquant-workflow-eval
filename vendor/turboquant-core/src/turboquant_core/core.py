"""
TurboQuant integration for Qwen3.5-9B (probe-verified).

Key probe findings for TQ:
  - DeltaNet state is OPAQUE: Qwen3_5DynamicCache returns None for DeltaNet layers
  - Only 8 GatedAttn layers have KV cache: K/V shape [batch, 4 heads, seq, 256]
  - TQ can only compress the GatedAttn KV cache (not DeltaNet internal state)
  - KV head_dim is 256 (not 128), num_kv_heads is 4 (not 8)
"""

import math

import torch
import numpy as np
from dataclasses import dataclass
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Codebook registry
# ---------------------------------------------------------------------------

@dataclass
class TQCodebook:
    dimension: int
    bit_width: int
    centroids: torch.Tensor
    boundaries: torch.Tensor
    mse_per_coord: float


def _lloyd_max_gaussian(b: int, max_iter: int = 200, tol: float = 1e-12):
    """Compute Lloyd-Max optimal quantizer for the standard normal distribution.

    Returns (centroids, mse_per_coord) where centroids is a sorted numpy array
    of 2^b reproduction levels minimizing E[(X - Q(X))^2] for X ~ N(0,1).
    """
    n_levels = 1 << b
    # Initialize centroids at evenly-spaced quantiles
    quantiles = np.linspace(0, 1, n_levels + 2)[1:-1]
    centroids = norm.ppf(quantiles)

    for _ in range(max_iter):
        # Boundaries = midpoints between adjacent centroids
        boundaries = np.concatenate([[-np.inf],
                                     (centroids[:-1] + centroids[1:]) / 2.0,
                                     [np.inf]])
        # Update centroids as conditional expectations E[X | b_i < X < b_{i+1}]
        new_centroids = np.empty(n_levels)
        for i in range(n_levels):
            lo, hi = boundaries[i], boundaries[i + 1]
            # E[X | lo < X < hi] = (phi(lo) - phi(hi)) / (Phi(hi) - Phi(lo))
            p = norm.cdf(hi) - norm.cdf(lo)
            if p < 1e-15:
                new_centroids[i] = (lo + hi) / 2.0
            else:
                new_centroids[i] = (norm.pdf(lo) - norm.pdf(hi)) / p

        if np.max(np.abs(new_centroids - centroids)) < tol:
            centroids = new_centroids
            break
        centroids = new_centroids

    # Compute MSE per coordinate: E[(X - Q(X))^2]
    boundaries = np.concatenate([[-np.inf],
                                 (centroids[:-1] + centroids[1:]) / 2.0,
                                 [np.inf]])
    mse = 0.0
    for i in range(n_levels):
        lo, hi = boundaries[i], boundaries[i + 1]
        p = norm.cdf(hi) - norm.cdf(lo)
        if p < 1e-15:
            continue
        c = centroids[i]
        # E[X|lo<X<hi] * P = phi(lo) - phi(hi), but x*phi(x)->0 as x->±inf
        phi_lo = norm.pdf(lo) if np.isfinite(lo) else 0.0
        phi_hi = norm.pdf(hi) if np.isfinite(hi) else 0.0
        ex_p = phi_lo - phi_hi
        # E[X^2|lo<X<hi]*P = P + lo*phi(lo) - hi*phi(hi)
        lo_phi_lo = lo * phi_lo if np.isfinite(lo) else 0.0
        hi_phi_hi = hi * phi_hi if np.isfinite(hi) else 0.0
        ex2_p = p + lo_phi_lo - hi_phi_hi
        mse += ex2_p - 2 * c * ex_p + c * c * p

    centroids.sort()
    return centroids, float(mse)


class CodebookRegistry:
    _cache: dict = {}

    @classmethod
    def get(cls, d: int, b: int, device=torch.device("cpu")) -> TQCodebook:
        key = (d, b)
        if key not in cls._cache:
            raw_centroids, mse_per_coord = _lloyd_max_gaussian(b)
            # Scale for the rotated unit-vector distribution: each coordinate ~ N(0, 1/d)
            scale = 1.0 / (d ** 0.5)
            centroids = torch.tensor(raw_centroids * scale, dtype=torch.float32)
            boundaries = torch.empty(len(centroids) + 1)
            boundaries[0] = -1.0
            boundaries[-1] = 1.0
            for i in range(len(centroids) - 1):
                boundaries[i + 1] = (centroids[i] + centroids[i + 1]) / 2.0
            cls._cache[key] = TQCodebook(d, b, centroids, boundaries, mse_per_coord / d)

        cb = cls._cache[key]
        if device != torch.device("cpu"):
            return TQCodebook(cb.dimension, cb.bit_width,
                              cb.centroids.to(device), cb.boundaries.to(device), cb.mse_per_coord)
        return cb

    @classmethod
    def list_cached(cls) -> list[tuple[int, int]]:
        """Return cached (dimension, bit_width) pairs."""
        return list(cls._cache.keys())

    @classmethod
    def precompute(cls, d: int, b: int, device=torch.device("cpu")) -> TQCodebook:
        """Compute, cache, and return a codebook for the given dimension and bit width."""
        return cls.get(d, b, device)

    @classmethod
    def clear(cls):
        """Drop all cached codebooks."""
        cls._cache.clear()


# ---------------------------------------------------------------------------
# Rotation cache (fast Walsh-Hadamard transform)
# ---------------------------------------------------------------------------

def _fast_wht(x):
    """In-place fast Walsh-Hadamard transform. O(d log d), no matrix materialized.

    x shape: [..., d] where d must be a power of 2.
    Returns the WHT of x (unnormalized — caller divides by sqrt(d)).
    """
    d = x.shape[-1]
    h = 1
    while h < d:
        # Butterfly: for each pair of blocks of size h, compute sum and difference
        x = x.reshape(*x.shape[:-1], -1, 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2)
        x = x.reshape(*x.shape[:-3], -1)
        h *= 2
    return x


class RotationCache:
    _cache: dict = {}

    @classmethod
    def get(cls, d: int, seed: int = 42, device=torch.device("cpu")):
        key = (d, seed)
        if key not in cls._cache:
            rng = np.random.default_rng(seed)
            d_pad = 1 << (d - 1).bit_length()
            signs = torch.tensor(rng.choice([-1.0, 1.0], size=d_pad), dtype=torch.float32)
            cls._cache[key] = {"d": d, "d_padded": d_pad, "signs": signs}
        e = cls._cache[key]
        if device != torch.device("cpu"):
            return {"d": e["d"], "d_padded": e["d_padded"],
                    "signs": e["signs"].to(device)}
        return e


# ---------------------------------------------------------------------------
# Core quantization
# ---------------------------------------------------------------------------

def tq_rotate(x, rot):
    d, dp, signs = rot["d"], rot["d_padded"], rot["signs"]
    if x.shape[-1] < dp:
        x = torch.nn.functional.pad(x, (0, dp - x.shape[-1]))
    return (_fast_wht(x * signs) / (dp ** 0.5))[..., :d]

def tq_rotate_inv(y, rot):
    d, dp, signs = rot["d"], rot["d_padded"], rot["signs"]
    if y.shape[-1] < dp:
        y = torch.nn.functional.pad(y, (0, dp - y.shape[-1]))
    return (_fast_wht(y) / (dp ** 0.5) * signs)[..., :d]

def tq_quantize_mse(x, cb, rot):
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    y = tq_rotate(x / (norms + 1e-12), rot)
    y = y.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
    indices = torch.searchsorted(cb.boundaries[1:-1].contiguous(), y.contiguous()).to(torch.uint8)
    return indices, norms.squeeze(-1)

def tq_dequantize_mse(indices, norms, cb, rot):
    y_hat = cb.centroids[indices.long()]
    return tq_rotate_inv(y_hat, rot) * norms.unsqueeze(-1)


# ---------------------------------------------------------------------------
# QJL
# ---------------------------------------------------------------------------

class QJLProjection:
    def __init__(self, d, seed=123, device=torch.device("cpu")):
        self.d = d
        # torch.Generator() defaults to CPU; PyTorch rejects a CPU
        # generator when the target tensor lives on CUDA. Build the
        # generator on the tensor's device so both CPU and CUDA paths
        # seed deterministically without the device-mismatch error.
        gen = torch.Generator(device=device).manual_seed(seed)
        self.S = torch.randn(d, d, generator=gen, device=device)

    def quantize(self, x):
        return (x @ self.S.T).sign().to(torch.int8)

    def to(self, device):
        self.S = self.S.to(device)
        return self


def tq_quantize_prod(x, cb, rot, qjl):
    norms = torch.linalg.norm(x, dim=-1)
    x_unit = x / (norms.unsqueeze(-1) + 1e-12)
    mse_idx, _ = tq_quantize_mse(x_unit, cb, rot)
    x_hat = tq_dequantize_mse(mse_idx, torch.ones_like(norms), cb, rot)
    r = x_unit - x_hat
    r_norms = torch.linalg.norm(r, dim=-1)
    qjl_bits = qjl.quantize(r / (r_norms.unsqueeze(-1) + 1e-12))
    return mse_idx, qjl_bits, r_norms, norms


# ---------------------------------------------------------------------------
# High-level wrappers
# ---------------------------------------------------------------------------

class TQActivationCheckpoint:
    def __init__(self, d, bit_width=4, seed=42, device=torch.device("cpu")):
        self.cb = CodebookRegistry.get(d, bit_width, device)
        self.rot = RotationCache.get(d, seed, device)
        self._idx = self._norms = self._shape = None

    def save(self, x):
        self._shape = x.shape
        flat = x.reshape(-1, x.shape[-1])
        self._idx, self._norms = tq_quantize_mse(flat, self.cb, self.rot)

    def restore(self):
        return tq_dequantize_mse(self._idx, self._norms, self.cb, self.rot).reshape(self._shape)


class TQLoRAStorage:
    def __init__(self, d_in, d_out, rank, bit_width=4, seed=42, device=torch.device("cpu")):
        self.d_in, self.d_out, self.rank = d_in, d_out, rank
        self.cb_in = CodebookRegistry.get(d_in, bit_width, device)
        self.cb_out = CodebookRegistry.get(d_out, bit_width, device)
        self.rot_in = RotationCache.get(d_in, seed, device)
        self.rot_out = RotationCache.get(d_out, seed + 1, device)

    def compress(self, A, B):
        Ai, An = tq_quantize_mse(A, self.cb_in, self.rot_in)
        Bi, Bn = tq_quantize_mse(B.T, self.cb_out, self.rot_out)
        return {"A_idx": Ai, "A_n": An, "B_idx": Bi, "B_n": Bn}

    def decompress(self, s):
        A = tq_dequantize_mse(s["A_idx"], s["A_n"], self.cb_in, self.rot_in)
        B = tq_dequantize_mse(s["B_idx"], s["B_n"], self.cb_out, self.rot_out).T
        return A, B


# ---------------------------------------------------------------------------
# Hybrid KV cache compression (probe-verified)
# ---------------------------------------------------------------------------

class TQGatedAttnKVCache:
    """
    Compresses KV cache for the 8 GatedAttn layers in Qwen3.5-9B.

    Probe findings:
      - Qwen3_5DynamicCache.key_cache[i] is None for DeltaNet layers
      - For GatedAttn layers: K/V shape [batch, 4 heads, seq_len, 256]
      - Only 8 of 32 layers have compressible KV cache

    Strategy:
      K → TQ_prod (MSE + QJL) — K participates in softmax(QK^T)
      V → TQ_MSE only — V is weighted-averaged by attention scores
    """

    def __init__(self, num_layers=32, interval=4,
                 kv_head_dim=256, num_kv_heads=4,
                 bit_width=4, seed=42, device=torch.device("cpu")):
        self.ga_indices = {i for i in range(num_layers) if (i + 1) % interval == 0}
        self.kv_head_dim = kv_head_dim

        # K: (b-1)-bit MSE + 1-bit QJL
        self.k_cb = CodebookRegistry.get(kv_head_dim, bit_width - 1, device)
        self.k_rot = RotationCache.get(kv_head_dim, seed, device)
        self.k_qjl = QJLProjection(kv_head_dim, seed=seed + 50, device=device)

        # V: b-bit MSE only (no QJL — V is not inner-producted against Q)
        self.v_cb = CodebookRegistry.get(kv_head_dim, bit_width, device)
        self.v_rot = RotationCache.get(kv_head_dim, seed + 100, device)

    def is_gated_attn(self, layer_idx):
        return layer_idx in self.ga_indices

    def compress_layer(self, K, V, layer_idx):
        """K, V shape: [batch, num_heads, seq_len, head_dim]"""
        assert self.is_gated_attn(layer_idx)
        b, nh, sl, hd = K.shape
        Kf = K.reshape(b * nh * sl, hd)
        Vf = V.reshape(b * nh * sl, hd)

        k_mse, k_qjl, k_rnorms, k_norms = tq_quantize_prod(Kf, self.k_cb, self.k_rot, self.k_qjl)
        v_idx, v_norms = tq_quantize_mse(Vf, self.v_cb, self.v_rot)

        return {"k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rnorms, "k_n": k_norms,
                "v_idx": v_idx, "v_n": v_norms, "shape": K.shape}

    def decompress_v(self, compressed):
        """Decompress V (MSE-only) for attention output computation."""
        s = compressed["shape"]
        Vf = tq_dequantize_mse(compressed["v_idx"], compressed["v_n"], self.v_cb, self.v_rot)
        return Vf.reshape(s)

    def compute_attention_scores(self, Q, compressed):
        """Compute unbiased Q @ K^T from fresh Q and compressed K using QJL correction.

        Q shape: [batch, num_heads, q_len, head_dim]
        Returns: [batch, num_heads, q_len, kv_len]
        """
        b, nh, kv_len, hd = compressed["shape"]
        q_len = Q.shape[2]

        # Stage 1: Q @ K_mse^T (biased, from MSE reconstruction)
        K_mse = tq_dequantize_mse(
            compressed["k_mse"], compressed["k_n"], self.k_cb, self.k_rot
        ).reshape(compressed["shape"])
        scores_mse = Q @ K_mse.transpose(-2, -1)

        # Stage 2: QJL bias correction on the residual
        # Reshape to [b*nh, seq, hd] for per-head batched matmul
        Q_flat = Q.reshape(b * nh * q_len, hd)
        Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
        k_qjl = compressed["k_qjl"].reshape(b * nh, kv_len, hd)

        # [b*nh, q_len, hd] @ [b*nh, hd, kv_len] -> [b*nh, q_len, kv_len]
        correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
        correction = correction * (math.pi / (2 * hd))
        # k_rn and k_n are [b*nh*kv_len] -> [b*nh, 1, kv_len] for broadcasting
        correction = correction * compressed["k_rn"].reshape(b * nh, 1, kv_len)
        correction = correction * compressed["k_n"].reshape(b * nh, 1, kv_len)
        correction = correction.reshape(b, nh, q_len, kv_len)

        return scores_mse + correction


def _resolve_head_indices(user_value, num_kv_heads):
    """Validate a ``compressible_heads`` list and return
    ``(frozenset | None, sorted_mask_tuple, complement_tuple)``.

    ``None`` means "compress every head" — returns ``(None, (), ())``.
    Any provided value must be a non-empty, duplicate-free list of ints
    in ``[0, num_kv_heads)``.
    """
    if user_value is None:
        return None, (), ()
    seen = set()
    duplicates = []
    bad_type = []
    bad_range = []
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
            "TQQuantizedCache.compressible_heads must contain ints; "
            f"got non-int entries {bad_type!r}."
        )
    if bad_range:
        raise ValueError(
            f"TQQuantizedCache.compressible_heads indices {sorted(set(bad_range))} "
            f"are out of range [0, {num_kv_heads})."
        )
    if duplicates:
        raise ValueError(
            "TQQuantizedCache.compressible_heads contains duplicate "
            f"indices {sorted(set(duplicates))}."
        )
    if not seen:
        raise ValueError(
            "TQQuantizedCache.compressible_heads must be non-empty when provided."
        )
    mask = tuple(sorted(seen))
    complement = tuple(h for h in range(num_kv_heads) if h not in seen)
    return frozenset(seen), mask, complement


# ---------------------------------------------------------------------------
# TQQuantizedCache — compressed KV storage with corrected attention
# ---------------------------------------------------------------------------

class TQQuantizedCache:
    """Compressed KV cache with configurable quantization strategy.

    Supports Qwen3.5 (hybrid GatedAttn/DeltaNet), Qwen3 (dense), and
    Qwen2.5 (dense) architectures. For non-compressible layers (DeltaNet),
    stores K/V in original precision.

    Key features:
      - key_strategy="mse": Full b-bit MSE codebook for keys (no QJL).
        Community consensus finds this outperforms MSE+QJL through softmax.
      - key_strategy="mse+qjl": (b-1)-bit MSE + 1-bit QJL residual
        correction (original paper approach).
      - residual_window=N: Keep the most recent N tokens in FP16,
        compress only older tokens. Critical for generation quality.
      - Causal masking: compute_attention() accepts causal_mask and
        attention_mask for correct multi-token prefill semantics.
    """

    def __init__(self, num_layers=32, interval=4,
                 kv_head_dim=256, num_kv_heads=4,
                 bit_width=4, seed=42, device=torch.device("cpu"),
                 residual_window=0, key_strategy="mse+qjl",
                 value_strategy="mse",
                 compressible_layers=None,
                 compressible_heads=None):
        self.num_layers = num_layers
        default_ga = {i for i in range(num_layers) if (i + 1) % interval == 0}
        if compressible_layers is None:
            self.ga_indices = default_ga
        else:
            # Validate and accept any non-empty subset of [0, num_layers).
            # Backend-level subset-of-default constraints (e.g. Qwen3.5
            # DeltaNet exclusion) are enforced at the backend layer; the
            # cache itself is permissive so dense backends can pass any
            # index list through.
            indices = set()
            bad: list[int] = []
            for idx in compressible_layers:
                if isinstance(idx, bool) or not isinstance(idx, int):
                    raise ValueError(
                        f"TQQuantizedCache.compressible_layers must contain ints; "
                        f"got {idx!r}"
                    )
                if idx < 0 or idx >= num_layers:
                    bad.append(idx)
                    continue
                indices.add(idx)
            if bad:
                raise ValueError(
                    f"TQQuantizedCache.compressible_layers indices {sorted(set(bad))} "
                    f"are out of range [0, {num_layers})."
                )
            if not indices:
                raise ValueError(
                    "TQQuantizedCache.compressible_layers must be non-empty when provided."
                )
            self.ga_indices = indices
        self.compressible_layers = sorted(self.ga_indices)
        self.head_indices, self._head_mask, self._head_complement = \
            _resolve_head_indices(compressible_heads, num_kv_heads)
        self.per_head_enabled = self.head_indices is not None
        self.compressible_heads = (
            list(self._head_mask) if self.per_head_enabled else None
        )
        self.kv_head_dim = kv_head_dim
        self.num_kv_heads = num_kv_heads
        self.device = device
        self.residual_window = residual_window
        self.key_strategy = key_strategy
        if value_strategy != "mse":
            raise ValueError(
                f"Invalid value_strategy {value_strategy!r}. "
                "Only 'mse' is currently supported for value compression."
            )
        self.value_strategy = value_strategy

        if key_strategy == "mse+qjl":
            # K: (b-1)-bit MSE + 1-bit QJL
            self.k_cb = CodebookRegistry.get(kv_head_dim, bit_width - 1, device)
            self.k_qjl = QJLProjection(kv_head_dim, seed=seed + 50, device=device)
        elif key_strategy == "mse":
            # K: b-bit MSE only (no QJL)
            self.k_cb = CodebookRegistry.get(kv_head_dim, bit_width, device)
            self.k_qjl = None
        else:
            raise ValueError(
                f"Invalid key_strategy {key_strategy!r}. Must be 'mse' or 'mse+qjl'."
            )
        self.k_rot = RotationCache.get(kv_head_dim, seed, device)

        # V: b-bit MSE only
        self.v_cb = CodebookRegistry.get(kv_head_dim, bit_width, device)
        self.v_rot = RotationCache.get(kv_head_dim, seed + 100, device)

        # Per-layer storage: compressed dicts for GA layers, raw tensors for others
        self._cache = [None] * num_layers
        self._seq_lens = [0] * num_layers
        # Residual window: recent tokens kept in FP16 per compressible layer
        self._window_k = [None] * num_layers
        self._window_v = [None] * num_layers

    def is_compressible(self, layer_idx):
        return layer_idx in self.ga_indices

    def _compress_kv(self, K_flat, V_flat, batch, num_heads, head_dim):
        """Compress flat K/V tensors according to key_strategy.

        Returns a dict with compressed representation.
        """
        v_idx, v_n = tq_quantize_mse(V_flat, self.v_cb, self.v_rot)
        if self.key_strategy == "mse+qjl":
            k_mse, k_qjl, k_rn, k_n = tq_quantize_prod(
                K_flat, self.k_cb, self.k_rot, self.k_qjl,
            )
            return {
                "k_mse": k_mse, "k_qjl": k_qjl, "k_rn": k_rn, "k_n": k_n,
                "v_idx": v_idx, "v_n": v_n,
                "batch": batch, "num_heads": num_heads, "head_dim": head_dim,
            }
        else:
            k_idx, k_n = tq_quantize_mse(K_flat, self.k_cb, self.k_rot)
            return {
                "k_mse": k_idx, "k_n": k_n,
                "v_idx": v_idx, "v_n": v_n,
                "batch": batch, "num_heads": num_heads, "head_dim": head_dim,
            }

    def _merge_compressed(self, old, new):
        """Concatenate two compressed cache entries.

        Flat quantized tensors (k_mse, k_qjl, k_rn, k_n, v_idx, v_n) are
        concatenated along dim 0 (the flattened ``b*nh*sl`` axis).
        Per-head passthrough buffers (``k_raw``/``v_raw``) are 4-D
        ``[b, len(complement), sl, hd]`` tensors and must concatenate on
        the sequence axis (dim 2) instead. Scalar metadata and the head
        mask tuples carry through unchanged; head_mask / head_complement /
        full_num_heads are asserted to match between old and new.
        """
        _RAW_KEYS = {"k_raw", "v_raw"}
        _MASK_KEYS = {"head_mask", "head_complement", "full_num_heads"}
        merged = {}
        for key in old:
            old_v = old[key]
            new_v = new.get(key, old_v)
            if key in _RAW_KEYS and isinstance(old_v, torch.Tensor):
                merged[key] = torch.cat([old_v, new_v], dim=2)
            elif isinstance(old_v, torch.Tensor):
                merged[key] = torch.cat([old_v, new_v], dim=0)
            elif key in _MASK_KEYS:
                if old_v != new_v:
                    raise ValueError(
                        f"TQQuantizedCache._merge_compressed: {key} drift "
                        f"(old={old_v!r}, new={new_v!r})"
                    )
                merged[key] = old_v
            else:
                merged[key] = new_v
        return merged

    def _compress_slice(self, K_slice, V_slice):
        """Compress a 4-D K/V slice ``[b, nh, sl, hd]`` and attach head-mask
        metadata if ``self.per_head_enabled``. Returns an entry dict.

        When per-head masking is active, only the ``self._head_mask`` heads
        go through the MSE/QJL quantizer; the complement heads ride along
        in full precision via ``k_raw``/``v_raw``.
        """
        b, nh, sl, hd = K_slice.shape
        if not self.per_head_enabled:
            Kf = K_slice.reshape(b * nh * sl, hd)
            Vf = V_slice.reshape(b * nh * sl, hd)
            return self._compress_kv(Kf, Vf, b, nh, hd)

        mask_idx = torch.tensor(self._head_mask, device=K_slice.device, dtype=torch.long)
        K_mask = K_slice.index_select(1, mask_idx).contiguous()
        V_mask = V_slice.index_select(1, mask_idx).contiguous()
        if self._head_complement:
            compl_idx = torch.tensor(
                self._head_complement, device=K_slice.device, dtype=torch.long
            )
            K_raw = K_slice.index_select(1, compl_idx).contiguous()
            V_raw = V_slice.index_select(1, compl_idx).contiguous()
        else:
            K_raw = K_slice.new_zeros(b, 0, sl, hd)
            V_raw = V_slice.new_zeros(b, 0, sl, hd)

        m = len(self._head_mask)
        Kf = K_mask.reshape(b * m * sl, hd)
        Vf = V_mask.reshape(b * m * sl, hd)
        entry = self._compress_kv(Kf, Vf, b, m, hd)
        entry["head_mask"] = self._head_mask
        entry["head_complement"] = self._head_complement
        entry["full_num_heads"] = nh
        entry["k_raw"] = K_raw
        entry["v_raw"] = V_raw
        return entry

    def update(self, K, V, layer_idx):
        """Store K/V for a layer. Compresses GatedAttn layers, stores raw otherwise.

        When residual_window > 0, the most recent `residual_window` tokens per
        compressible layer are kept in full FP16 precision. Older tokens that
        fall outside the window are compressed and moved to the quantized cache.

        K, V shape: [batch, num_heads, seq_len, head_dim]
        """
        if not self.is_compressible(layer_idx):
            # DeltaNet or non-compressible: store raw
            if self._cache[layer_idx] is None:
                self._cache[layer_idx] = (K, V)
            else:
                old_k, old_v = self._cache[layer_idx]
                self._cache[layer_idx] = (
                    torch.cat([old_k, K], dim=2),
                    torch.cat([old_v, V], dim=2),
                )
            self._seq_lens[layer_idx] = self._cache[layer_idx][0].shape[2]
            return

        b, nh, sl, hd = K.shape

        # Append new tokens to the FP16 residual window
        if self._window_k[layer_idx] is None:
            window_k = K
            window_v = V
        else:
            window_k = torch.cat([self._window_k[layer_idx], K], dim=2)
            window_v = torch.cat([self._window_v[layer_idx], V], dim=2)

        rw = self.residual_window
        if rw > 0 and window_k.shape[2] > rw:
            # Tokens that have aged out of the window → compress them
            overflow_len = window_k.shape[2] - rw
            to_compress_k = window_k[:, :, :overflow_len, :]
            to_compress_v = window_v[:, :, :overflow_len, :]

            # Keep the recent rw tokens in FP16
            self._window_k[layer_idx] = window_k[:, :, overflow_len:, :]
            self._window_v[layer_idx] = window_v[:, :, overflow_len:, :]

            new_entry = self._compress_slice(to_compress_k, to_compress_v)

            if self._cache[layer_idx] is None:
                self._cache[layer_idx] = new_entry
            else:
                self._cache[layer_idx] = self._merge_compressed(
                    self._cache[layer_idx], new_entry,
                )

            self._seq_lens[layer_idx] = (
                self._compressed_seq_len(layer_idx) + self._window_k[layer_idx].shape[2]
            )
        elif rw > 0:
            # All tokens fit in the window — no compression yet
            self._window_k[layer_idx] = window_k
            self._window_v[layer_idx] = window_v
            self._seq_lens[layer_idx] = window_k.shape[2]
        else:
            # No residual window: compress everything immediately
            self._window_k[layer_idx] = None
            self._window_v[layer_idx] = None

            new_entry = self._compress_slice(K, V)

            if self._cache[layer_idx] is None:
                self._cache[layer_idx] = new_entry
                self._seq_lens[layer_idx] = sl
            else:
                self._cache[layer_idx] = self._merge_compressed(
                    self._cache[layer_idx], new_entry,
                )
                self._seq_lens[layer_idx] += sl

    def _compressed_seq_len(self, layer_idx):
        """Return the number of tokens in the compressed cache for a layer."""
        entry = self._cache[layer_idx]
        if entry is None:
            return 0
        b = entry["batch"]
        nh = entry["num_heads"]  # masked head count when per-head is active
        total_flat = entry["k_mse"].shape[0]
        return total_flat // (b * nh)

    def _reconstruct_full_kv(self, entry):
        """Reassemble full-head K/V tensors from a compressed entry when
        per-head masking was applied. Returns ``(K_full, V_full)`` of shape
        ``[b, full_num_heads, compressed_len, head_dim]``.

        Masked heads come from the MSE-dequantized quantizer output;
        complement heads come from the raw passthrough buffers. QJL bias
        correction is deliberately skipped on this path — see the module
        docstring.
        """
        b = entry["batch"]
        m = entry["num_heads"]
        hd = entry["head_dim"]
        nh_full = entry["full_num_heads"]
        compressed_len = entry["k_mse"].shape[0] // (b * m)
        mask = list(entry["head_mask"])
        complement = list(entry["head_complement"])

        K_mse = tq_dequantize_mse(
            entry["k_mse"], entry["k_n"], self.k_cb, self.k_rot,
        ).reshape(b, m, compressed_len, hd)
        V_comp = tq_dequantize_mse(
            entry["v_idx"], entry["v_n"], self.v_cb, self.v_rot,
        ).reshape(b, m, compressed_len, hd)

        K_full = K_mse.new_empty(b, nh_full, compressed_len, hd)
        V_full = V_comp.new_empty(b, nh_full, compressed_len, hd)
        K_full[:, mask, :, :] = K_mse
        V_full[:, mask, :, :] = V_comp
        if complement:
            # k_raw / v_raw preserve the model's original dtype (e.g. bfloat16),
            # while K_mse / V_comp come out of tq_dequantize_mse as float32.
            # Cast the passthrough slices to K_full's dtype so the index_put
            # doesn't raise "source and destination dtypes match" on mixed
            # bf16 / fp32 models.
            K_full[:, complement, :, :] = entry["k_raw"].to(K_full.dtype)
            V_full[:, complement, :, :] = entry["v_raw"].to(V_full.dtype)
        return K_full, V_full

    def get_seq_length(self, layer_idx=0):
        return self._seq_lens[layer_idx]

    def compute_attention(self, Q, layer_idx, causal_mask=None, attention_mask=None):
        """Compute attention output: softmax(Q @ K^T / sqrt(d)) @ V.

        For compressed layers, uses QJL-corrected attention scores.
        For raw layers, uses standard attention.
        When residual_window > 0, combines compressed scores with FP16 window scores.

        Args:
            Q: [batch, num_heads, q_len, head_dim]
            layer_idx: Layer index.
            causal_mask: Optional [1, 1, q_len, kv_len] boolean mask (True = attend).
            attention_mask: Optional HF-style mask [batch, 1, q_len/1, kv_len]
                with 0 for valid, large negative for masked positions.

        Returns: [batch, num_heads, q_len, head_dim]
        """
        if not self.is_compressible(layer_idx):
            entry = self._cache[layer_idx]
            if entry is None:
                raise ValueError(f"No cache for layer {layer_idx}")
            K, V = entry
            scores = Q @ K.transpose(-2, -1) / (Q.shape[-1] ** 0.5)
            if causal_mask is not None:
                scores = scores.masked_fill(~causal_mask, float("-inf"))
            if attention_mask is not None:
                scores = scores + attention_mask
            attn = torch.softmax(scores, dim=-1)
            return attn @ V

        entry = self._cache[layer_idx]
        has_compressed = entry is not None
        has_window = self._window_k[layer_idx] is not None

        if not has_compressed and not has_window:
            raise ValueError(f"No cache for layer {layer_idx}")

        q_len = Q.shape[2]
        hd = Q.shape[3]

        # --- Window-only path (all tokens in FP16) ---
        if not has_compressed and has_window:
            wk = self._window_k[layer_idx]
            wv = self._window_v[layer_idx]
            scores = Q @ wk.transpose(-2, -1) / (hd ** 0.5)
            if causal_mask is not None:
                scores = scores.masked_fill(~causal_mask, float("-inf"))
            if attention_mask is not None:
                scores = scores + attention_mask
            attn = torch.softmax(scores, dim=-1)
            return attn @ wv

        # --- Compressed path (possibly with residual window) ---
        compressed_len = self._compressed_seq_len(layer_idx)

        if "head_mask" in entry:
            # Per-head masking: reconstruct full-head K/V (MSE-dequantized
            # for masked heads, raw for complement heads). QJL bias
            # correction is applied to the masked heads only and scattered
            # into a zero-filled full-head buffer before summing — see
            # the masked/full scatter idiom in _reconstruct_full_kv.
            K_full, V_compressed = self._reconstruct_full_kv(entry)
            scores_mse = Q @ K_full.transpose(-2, -1)
            if self.key_strategy == "mse+qjl":
                b = entry["batch"]
                m = entry["num_heads"]             # masked head count
                nh_full = entry["full_num_heads"]
                mask = list(entry["head_mask"])
                Q_m = Q.index_select(
                    1, torch.tensor(mask, device=Q.device, dtype=torch.long)
                )
                Q_flat = Q_m.reshape(b * m * q_len, hd)
                Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * m, q_len, hd)
                k_qjl = entry["k_qjl"].reshape(b * m, compressed_len, hd)
                correction_m = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
                correction_m = correction_m * (math.pi / (2 * hd))
                correction_m = correction_m * entry["k_rn"].reshape(b * m, 1, compressed_len)
                correction_m = correction_m * entry["k_n"].reshape(b * m, 1, compressed_len)
                correction_m = correction_m.reshape(b, m, q_len, compressed_len)
                correction_full = correction_m.new_zeros(
                    b, nh_full, q_len, compressed_len
                )
                correction_full[:, mask, :, :] = correction_m
                compressed_scores = (scores_mse + correction_full) / (hd ** 0.5)
            else:
                compressed_scores = scores_mse / (hd ** 0.5)
        else:
            b = entry["batch"]
            nh = entry["num_heads"]
            c_shape = (b, nh, compressed_len, hd)

            # MSE-reconstructed K
            K_mse = tq_dequantize_mse(
                entry["k_mse"], entry["k_n"], self.k_cb, self.k_rot
            ).reshape(c_shape)
            scores_mse = Q @ K_mse.transpose(-2, -1)

            # QJL bias correction (only if key_strategy == "mse+qjl")
            if self.key_strategy == "mse+qjl":
                Q_flat = Q.reshape(b * nh * q_len, hd)
                Q_qjl = self.k_qjl.quantize(Q_flat).reshape(b * nh, q_len, hd)
                k_qjl = entry["k_qjl"].reshape(b * nh, compressed_len, hd)

                correction = Q_qjl.float() @ k_qjl.float().transpose(-2, -1)
                correction = correction * (math.pi / (2 * hd))
                correction = correction * entry["k_rn"].reshape(b * nh, 1, compressed_len)
                correction = correction * entry["k_n"].reshape(b * nh, 1, compressed_len)
                correction = correction.reshape(b, nh, q_len, compressed_len)

                compressed_scores = (scores_mse + correction) / (hd ** 0.5)
            else:
                compressed_scores = scores_mse / (hd ** 0.5)

            # Decompress V from compressed region
            V_compressed = tq_dequantize_mse(
                entry["v_idx"], entry["v_n"], self.v_cb, self.v_rot
            ).reshape(c_shape)

        if has_window:
            # Combine compressed scores with FP16 window scores
            wk = self._window_k[layer_idx]
            wv = self._window_v[layer_idx]
            window_scores = Q @ wk.transpose(-2, -1) / (hd ** 0.5)

            # Concatenate: [compressed_scores | window_scores]
            all_scores = torch.cat([compressed_scores, window_scores], dim=-1)
            # Concatenate: [V_compressed | V_window]
            all_V = torch.cat([V_compressed, wv], dim=2)
        else:
            all_scores = compressed_scores
            all_V = V_compressed

        if causal_mask is not None:
            all_scores = all_scores.masked_fill(~causal_mask, float("-inf"))
        if attention_mask is not None:
            all_scores = all_scores + attention_mask
        attn = torch.softmax(all_scores, dim=-1)
        return attn @ all_V

    def clear(self):
        self._cache = [None] * self.num_layers
        self._seq_lens = [0] * self.num_layers
        self._window_k = [None] * self.num_layers
        self._window_v = [None] * self.num_layers


# ---------------------------------------------------------------------------
# Straight-through estimator for differentiable quantization
# ---------------------------------------------------------------------------

class _STEQuantize(torch.autograd.Function):
    """Straight-through estimator: forward does hard quantization,
    backward passes gradients through as if quantization were identity."""

    @staticmethod
    def forward(ctx, y, boundaries, centroids):
        indices = torch.searchsorted(boundaries[1:-1].contiguous(), y.contiguous())
        return centroids[indices]

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def tq_quantize_mse_ste(x, cb, rot):
    """Differentiable MSE quantization using straight-through estimator.

    Forward: identical to tq_quantize_mse/tq_dequantize_mse (hard quantization).
    Backward: gradients pass through the quantization as if it were identity.

    Returns the reconstructed tensor (not indices), preserving the gradient graph.
    """
    norms = torch.linalg.norm(x, dim=-1, keepdim=True)
    x_unit = x / (norms + 1e-12)
    y = tq_rotate(x_unit, rot)
    y = y.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

    # STE: forward quantizes, backward is identity
    y_hat = _STEQuantize.apply(y, cb.boundaries, cb.centroids)

    return tq_rotate_inv(y_hat, rot) * norms
