#!/usr/bin/env python3
"""GQA/MQA smoke: KV savings, fidelity, matching task → results/."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from experiments import (
    compare_modes_trained,
    expand_roundtrip_check,
    random_weight_fidelity,
    structural_kv_report,
)
from smoke_plots import make_plots, write_results_md

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SEED = 42
CFG = {
    "d_model": 64,
    "n_q_heads": 8,
    "n_kv_gqa": 2,
    "seq_len": 16,
    "n_distractors": 2,
    "n_train": 640,
    "n_test": 160,
    "steps": 220,
    "lr": 0.18,
    "logit_scale": 12.0,
    "kv_batch": 1,
    "kv_seq_len": 2048,
    "kv_d_head": 64,
    "fidelity_seq_len": 32,
    "fidelity_batch": 4,
}


def _compact(js: str) -> str:
    js = re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]",
        js,
    )
    return re.sub(
        r"\{\n([^{}\[\]]*?)\n\s*\}",
        lambda m: "{" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "}",
        js,
    )


def main() -> None:
    t0 = time.perf_counter()
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.random.seed(SEED)

    expand = expand_roundtrip_check(
        CFG["n_q_heads"], CFG["n_kv_gqa"], seed=SEED
    )
    expand_mqa = expand_roundtrip_check(CFG["n_q_heads"], 1, seed=SEED)

    kv = structural_kv_report(
        CFG["n_q_heads"],
        CFG["n_kv_gqa"],
        batch=CFG["kv_batch"],
        seq_len=CFG["kv_seq_len"],
        d_head=CFG["kv_d_head"],
    )

    fid = random_weight_fidelity(
        d_model=CFG["d_model"],
        n_q_heads=CFG["n_q_heads"],
        n_kv_gqa=CFG["n_kv_gqa"],
        seq_len=CFG["fidelity_seq_len"],
        batch=CFG["fidelity_batch"],
        seed=SEED,
    )

    trained = compare_modes_trained(
        d_model=CFG["d_model"],
        seq_len=CFG["seq_len"],
        n_q_heads=CFG["n_q_heads"],
        n_kv_gqa=CFG["n_kv_gqa"],
        n_train=CFG["n_train"],
        n_test=CFG["n_test"],
        steps=CFG["steps"],
        lr=CFG["lr"],
        seed=SEED,
        n_distractors=CFG["n_distractors"],
        logit_scale=CFG["logit_scale"],
    )

    wall = round(time.perf_counter() - t0, 2)
    by = trained["by_mode"]
    headline = {
        "expand_pass": expand["pass"] and expand_mqa["pass"],
        "gqa_reduction_vs_mha": kv["headline"]["gqa_reduction_vs_mha"],
        "mqa_reduction_vs_mha": kv["headline"]["mqa_reduction_vs_mha"],
        "kv_ratio_checks_pass": kv["checks"]["gqa_ratio_matches_theory"]
        and kv["checks"]["mqa_ratio_matches_theory"],
        "cosine_gqa": fid["cosine_gqa_collapsed"],
        "cosine_mqa": fid["cosine_mqa_collapsed"],
        "gqa_closer_than_mqa": fid["gqa_closer_than_mqa"],
        "mha_acc": by["mha"]["final_acc"],
        "gqa_acc": by["gqa"]["final_acc"],
        "mqa_acc": by["mqa"]["final_acc"],
        "gqa_near_mha": trained["gqa_near_mha"],
        "gqa_beats_mqa": trained["gqa_beats_mqa"],
        "all_above_chance": trained["all_above_chance"],
    }

    trained_json = {
        "modes": [
            {
                "mode": r["mode"],
                "n_q_heads": r["n_q_heads"],
                "n_kv_heads": r["n_kv_heads"],
                "n_rep": r["n_rep"],
                "final_acc": r["final_acc"],
                "final_mass": r["final_mass"],
                "chance_acc": r["chance_acc"],
                "history": r["history"],
                "attn_row": r["attn_row"],
                "label": r["label"],
                "query_idx": r["query_idx"],
            }
            for r in trained["modes"]
        ],
        "by_mode": trained["by_mode"],
        "gqa_near_mha": trained["gqa_near_mha"],
        "gqa_beats_mqa": trained["gqa_beats_mqa"],
        "all_above_chance": trained["all_above_chance"],
    }

    metrics: Dict[str, Any] = {
        "project": "ai-learn-28-grouped-query-attention",
        "seed": SEED,
        "config": CFG,
        "expand_check": {"gqa": expand, "mqa": expand_mqa},
        "kv_savings": kv,
        "fidelity": fid,
        "trained": trained_json,
        "headline": headline,
        "wall_time_s": wall,
    }

    plot_names = make_plots(RESULTS, metrics)
    write_results_md(RESULTS, metrics, plot_names)

    (RESULTS / "metrics.json").write_text(
        _compact(json.dumps(metrics, indent=1)) + "\n", encoding="utf-8"
    )
    shot = {
        "project": "ai-learn-28-grouped-query-attention",
        "seed": SEED,
        "config": {
            "d_model": CFG["d_model"],
            "n_q_heads": CFG["n_q_heads"],
            "n_kv_gqa": CFG["n_kv_gqa"],
            "seq_len": CFG["seq_len"],
            "steps": CFG["steps"],
            "logit_scale": CFG["logit_scale"],
        },
        "headline": headline,
        "wall_time_s": wall,
    }
    (RESULTS / "JSON.shot").write_text(
        _compact(json.dumps(shot, indent=1)) + "\n", encoding="utf-8"
    )

    print("=== ai-learn-28-grouped-query-attention smoke ===")
    print(f"seed={SEED} wall_time_s={wall}")
    print(f"expand_pass={headline['expand_pass']}")
    print(
        f"KV reduction vs MHA: GQA={headline['gqa_reduction_vs_mha']}× "
        f"MQA={headline['mqa_reduction_vs_mha']}×"
    )
    print(
        f"cosine fidelity: GQA={headline['cosine_gqa']} MQA={headline['cosine_mqa']}"
    )
    for mode in ("mha", "gqa", "mqa"):
        r = by[mode]
        print(f"  {mode:3s} n_kv={r['n_kv_heads']} acc={r['final_acc']} mass={r['final_mass']}")
    print("plots:", ", ".join(plot_names))
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
