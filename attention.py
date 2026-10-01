"""Tiny NumPy multi-head attention with configurable n_heads_q / n_heads_kv.

Supports MHA (n_kv == n_q), GQA (1 < n_kv < n_q), and MQA (n_kv == 1).
K/V are projected with fewer heads and expanded via repeat_kv before the
scaled-dot-product attention — Llama-2 / Ainslie-style GQA.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from gqa import expand_kv_for_gqa, mode_from_heads, n_rep, validate_heads

Mode = Literal["mha", "gqa", "mqa"]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.where(np.isfinite(x), x, -1e9)
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


class GQAAttention:
    """Multi-head attention with grouped / multi-query K/V sharing.

    Parameters
    ----------
    d_model : model width (must be divisible by n_q_heads)
    n_q_heads : number of query heads
    n_kv_heads : number of key/value heads (divides n_q_heads)
    """

    def __init__(
        self,
        d_model: int,
        *,
        n_q_heads: int = 8,
        n_kv_heads: int | None = None,
        causal: bool = False,
        rng: np.random.Generator | None = None,
        scale: float = 0.1,
    ) -> None:
        if d_model < 2 or d_model % 2 != 0:
            raise ValueError("d_model must be even and >= 2")
        if n_kv_heads is None:
            n_kv_heads = n_q_heads
        if d_model % n_q_heads != 0:
            raise ValueError("d_model must be divisible by n_q_heads")
        validate_heads(n_q_heads, n_kv_heads)

        self.d_model = d_model
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.d_head = d_model // n_q_heads
        self.n_rep = n_rep(n_q_heads, n_kv_heads)
        self.mode: Mode = mode_from_heads(n_q_heads, n_kv_heads)
        self.causal = causal

        rng = rng or np.random.default_rng()
        # Q: full width; K/V: only n_kv_heads * d_head columns
        self.W_q = rng.normal(0, scale, size=(d_model, n_q_heads * self.d_head))
        self.W_k = rng.normal(0, scale, size=(d_model, n_kv_heads * self.d_head))
        self.W_v = rng.normal(0, scale, size=(d_model, n_kv_heads * self.d_head))
        self.W_o = rng.normal(0, scale, size=(n_q_heads * self.d_head, d_model))
        self.b_o = np.zeros(d_model, dtype=np.float64)

    def parameters(self) -> list[np.ndarray]:
        return [self.W_q, self.W_k, self.W_v, self.W_o, self.b_o]

    def _split_q(self, t: np.ndarray) -> np.ndarray:
        # (B, T, n_q*Dh) -> (B, n_q, T, Dh)
        B, T, _ = t.shape
        return t.reshape(B, T, self.n_q_heads, self.d_head).transpose(0, 2, 1, 3)

    def _split_kv(self, t: np.ndarray) -> np.ndarray:
        # (B, T, n_kv*Dh) -> (B, n_kv, T, Dh)
        B, T, _ = t.shape
        return t.reshape(B, T, self.n_kv_heads, self.d_head).transpose(0, 2, 1, 3)

    def project_qkv(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Q (B,Hq,T,Dh), K/V (B,Hkv,T,Dh) before KV expand."""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, ...]
        Q = self._split_q(x @ self.W_q)
        K = self._split_kv(x @ self.W_k)
        V = self._split_kv(x @ self.W_v)
        return Q, K, V

    def qkv_logits(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Q, expanded K, expanded V, and attention logits (B, Hq, T, T)."""
        Q, K, V = self.project_qkv(x)
        K_exp, V_exp = expand_kv_for_gqa(K, V, self.n_q_heads, self.n_kv_heads)
        scale = 1.0 / np.sqrt(self.d_head)
        logits = (Q @ K_exp.transpose(0, 1, 3, 2)) * scale
        if self.causal:
            T = logits.shape[-1]
            mask = np.triu(np.ones((T, T), dtype=bool), k=1)
            logits = np.where(mask[None, None, :, :], -np.inf, logits)
        return Q, K_exp, V_exp, logits

    def forward(self, x: np.ndarray, *, return_weights: bool = False):
        Q, K_exp, V_exp, logits = self.qkv_logits(x)
        weights = softmax(logits, axis=-1)
        ctx = weights @ V_exp  # (B, Hq, T, Dh)
        B, H, T, Dh = ctx.shape
        ctx = ctx.transpose(0, 2, 1, 3).reshape(B, T, H * Dh)
        out = ctx @ self.W_o + self.b_o
        if return_weights:
            return out, weights, logits
        return out


def make_attention(
    d_model: int,
    mode: Mode,
    *,
    n_q_heads: int = 8,
    n_kv_gqa: int = 2,
    causal: bool = False,
    rng: np.random.Generator | None = None,
    scale: float = 0.1,
) -> GQAAttention:
    """Factory: mode ∈ {mha, gqa, mqa} → correct n_kv_heads."""
    if mode == "mha":
        n_kv = n_q_heads
    elif mode == "mqa":
        n_kv = 1
    elif mode == "gqa":
        n_kv = n_kv_gqa
    else:
        raise ValueError(f"unknown mode {mode!r}")
    return GQAAttention(
        d_model,
        n_q_heads=n_q_heads,
        n_kv_heads=n_kv,
        causal=causal,
        rng=rng,
        scale=scale,
    )


def clone_to_mode(src: GQAAttention, mode: Mode, *, n_kv_gqa: int = 2) -> GQAAttention:
    """Build a new attn in ``mode``; copy shared projection slices when possible.

    For fidelity experiments we usually *re-init* each mode independently.
    This helper copies W_q / W_o fully and copies the first n_kv KV columns
    from an MHA source (useful for collapse-from-MHA demos).
    """
    dst = make_attention(
        src.d_model,
        mode,
        n_q_heads=src.n_q_heads,
        n_kv_gqa=n_kv_gqa,
        causal=src.causal,
        scale=0.0,
    )
    dst.W_q = src.W_q.copy()
    dst.W_o = src.W_o.copy()
    dst.b_o = src.b_o.copy()
    # Copy as many KV head columns as dst has from src
    src_kv_dim = src.n_kv_heads * src.d_head
    dst_kv_dim = dst.n_kv_heads * dst.d_head
    take = min(src_kv_dim, dst_kv_dim)
    dst.W_k[:, :take] = src.W_k[:, :take]
    dst.W_v[:, :take] = src.W_v[:, :take]
    return dst
