"""Matplotlib SVG plots + RESULTS.md writer for GQA/MQA smoke."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from svg_utils import minify_svg  # noqa: E402

plt.rcParams.update(
    {
        "svg.hashsalt": "ai-learn-28",
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "axes.unicode_minus": False,
    }
)
COLS = ["#e76f51", "#e9c46a", "#2a9d8f", "#126782", "#8338ec"]
MODE_ORDER = ["mha", "gqa", "mqa"]
MODE_LABELS = {"mha": "MHA", "gqa": "GQA", "mqa": "MQA"}


def _save(fig, path: Path) -> str:
    fig.tight_layout()
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None})
    plt.close(fig)
    path.write_text(minify_svg(buf.getvalue()), encoding="utf-8")
    return path.name


def make_plots(out: Path, m: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    modes = m["kv_savings"]["modes"]

    # 1) KV cache bytes bars
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    xs = np.arange(len(MODE_ORDER))
    bytes_mib = [modes[k]["mib"] for k in MODE_ORDER]
    colors = [COLS[3], COLS[2], COLS[0]]
    bars = ax.bar(xs, bytes_mib, color=colors, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODE_LABELS[k] for k in MODE_ORDER])
    ax.set_ylabel("KV cache (MiB)")
    ax.set_title(
        f"KV cache size (B={m['kv_savings']['batch']}, "
        f"T={m['kv_savings']['seq_len']}, d_h={m['kv_savings']['d_head']}, fp32)"
    )
    for b, k in zip(bars, MODE_ORDER):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{modes[k]['vs_mha_ratio']:g}x",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "kv_cache_bytes.svg"))

    # 2) Reduction ratio bars
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ratios = [modes[k]["vs_mha_ratio"] for k in MODE_ORDER]
    ax.bar(xs, ratios, color=colors, width=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([MODE_LABELS[k] for k in MODE_ORDER])
    ax.set_ylabel("reduction vs MHA (x)")
    ax.set_title(
        f"KV reduction (Hq={m['config']['n_q_heads']}, "
        f"GQA Hkv={m['config']['n_kv_gqa']})"
    )
    ax.axhline(1.0, color="#555", ls="--", lw=1)
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "kv_reduction_ratio.svg"))

    # 3) Cosine fidelity vs MHA
    fid = m["fidelity"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    labels = ["GQA\n(collapsed KV)", "MQA\n(collapsed KV)", "GQA\n(random KV)"]
    vals = [
        fid["cosine_gqa_collapsed"],
        fid["cosine_mqa_collapsed"],
        fid["cosine_gqa_random_kv"],
    ]
    ax.bar(np.arange(3), vals, color=[COLS[2], COLS[0], COLS[1]], width=0.6)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean cosine vs MHA output")
    ax.set_title("Attention output fidelity vs full MHA (random weights)")
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "fidelity_cosine.svg"))

    # 4) Trained matching accuracy
    trained = m["trained"]["by_mode"]
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    accs = [trained[k]["final_acc"] for k in MODE_ORDER]
    masses = [trained[k]["final_mass"] for k in MODE_ORDER]
    w = 0.35
    ax.bar(xs - w / 2, accs, w, color=COLS[2], label="top-1 acc")
    ax.bar(xs + w / 2, masses, w, color=COLS[3], label="attn mass @ target")
    ax.set_xticks(xs)
    ax.set_xticklabels([MODE_LABELS[k] for k in MODE_ORDER])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("Planted-key matching after short train (Q/K)")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    names.append(_save(fig, out / "matching_accuracy.svg"))

    # 5) Training curves
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for mode, c in zip(MODE_ORDER, colors):
        block = next(r for r in m["trained"]["modes"] if r["mode"] == mode)
        steps = [h["step"] for h in block["history"]]
        accs_h = [h["acc"] for h in block["history"]]
        ax.plot(steps, accs_h, "o-", color=c, label=MODE_LABELS[mode])
    ax.set_xlabel("step")
    ax.set_ylabel("test top-1 acc")
    ax.set_ylim(0, 1.05)
    ax.set_title("Learning curves (matching task)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "learning_curves.svg"))

    # 6) Head grouping diagram-style: n_kv vs bytes
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    n_q = m["config"]["n_q_heads"]
    # Show continuum: Hkv = 1..Hq
    hkvs = list(range(1, n_q + 1))
    if n_q % 1 == 0:
        hkvs = [h for h in hkvs if n_q % h == 0]
    d_head = m["kv_savings"]["d_head"]
    seq_len = m["kv_savings"]["seq_len"]
    batch = m["kv_savings"]["batch"]
    y = [2 * batch * seq_len * h * d_head * 4 / (1024 * 1024) for h in hkvs]
    ax.plot(hkvs, y, "o-", color=COLS[3])
    for mode, c, marker in (("mha", COLS[3], "s"), ("gqa", COLS[2], "D"), ("mqa", COLS[0], "o")):
        hkv = modes[mode]["n_kv_heads"]
        ax.scatter([hkv], [modes[mode]["mib"]], color=c, s=80, zorder=5, marker=marker, label=MODE_LABELS[mode])
    ax.set_xlabel("n_kv_heads")
    ax.set_ylabel("KV cache (MiB)")
    ax.set_title(f"KV bytes vs n_kv_heads (Hq={n_q} fixed)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    names.append(_save(fig, out / "kv_vs_heads.svg"))

    return names


def write_results_md(out: Path, m: Dict[str, Any], plot_names: List[str]) -> None:
    h = m["headline"]
    modes = m["kv_savings"]["modes"]
    fid = m["fidelity"]
    trained = m["trained"]["by_mode"]
    lines = [
        "# Results: ai-learn-28-grouped-query-attention",
        "",
        f"Real output of `python run_smoke.py` (seed {m['seed']}, CPU, {m['wall_time_s']} s wall time).",
        "",
        "## Setup",
        f"- d_model={m['config']['d_model']}, n_q_heads={m['config']['n_q_heads']}, "
        f"n_kv_gqa={m['config']['n_kv_gqa']}, seq_len(train)={m['config']['seq_len']}.",
        f"- KV-cache table: B={m['kv_savings']['batch']}, T={m['kv_savings']['seq_len']}, "
        f"d_head={m['kv_savings']['d_head']}, dtype=float32.",
        f"- Matching task: {m['config']['n_distractors']} content-matched distractors; "
        f"train Q/K for {m['config']['steps']} steps, logit_scale={m['config']['logit_scale']}.",
        "",
        "## KV cache (structural)",
        "| mode | n_kv_heads | n_rep | bytes | MiB | vs MHA |",
        "|---|---|---|---|---|---|",
    ]
    for mode in MODE_ORDER:
        r = modes[mode]
        lines.append(
            f"| {MODE_LABELS[mode]} | {r['n_kv_heads']} | {r['n_rep']} | "
            f"{r['bytes']} | {r['mib']} | {r['vs_mha_ratio']}x |"
        )
    lines += [
        "",
        "## Output fidelity vs MHA (random weights, collapsed KV)",
        f"- GQA cosine: **{fid['cosine_gqa_collapsed']}**",
        f"- MQA cosine: **{fid['cosine_mqa_collapsed']}**",
        f"- GQA (random KV, shared Q/O): {fid['cosine_gqa_random_kv']}",
        f"- GQA closer than MQA: {fid['gqa_closer_than_mqa']}",
        "",
        "## Planted-key matching (after train)",
        "| mode | n_kv | top-1 acc | attn mass @ target |",
        "|---|---|---|---|",
    ]
    for mode in MODE_ORDER:
        block = next(r for r in m["trained"]["modes"] if r["mode"] == mode)
        lines.append(
            f"| {MODE_LABELS[mode]} | {block['n_kv_heads']} | "
            f"{block['final_acc']} | {block['final_mass']} |"
        )
    lines += [
        "",
        "## Headline",
        f"- KV reduction: GQA **{h['gqa_reduction_vs_mha']}x**, MQA **{h['mqa_reduction_vs_mha']}x** vs MHA.",
        f"- Cosine fidelity (collapsed): GQA **{h['cosine_gqa']}** > MQA **{h['cosine_mqa']}**.",
        f"- Matching acc: MHA **{h['mha_acc']}**, GQA **{h['gqa_acc']}**, MQA **{h['mqa_acc']}**.",
        f"- Expand round-trip check: pass={h['expand_pass']}.",
        "",
        "## Plots",
    ]
    for name in plot_names:
        lines.append(f"- ![{name}]({name})")
    lines += [
        "",
        "## Notes",
        "- GQA repeats each KV head `n_rep = Hq / Hkv` times before QK^T (Llama-2 style).",
        "- Collapsed-KV fidelity averages MHA's per-head K/V within groups - milder for GQA than MQA.",
        "- Toy NumPy attention + synthetic matching task, not a full Transformer / LM.",
        "",
    ]
    (out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
