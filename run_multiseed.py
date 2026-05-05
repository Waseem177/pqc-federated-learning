#!/usr/bin/env python3
"""
Multi-seed evaluation harness.
Runs simulation.py 5 times with different seeds, aggregates mean ± std,
and saves results_multiseed.json.
"""

import glob
import json
import subprocess
import sys
from datetime import datetime

import numpy as np

SEEDS  = [42, 123, 456, 789, 1337]
ROUNDS = 50
NODES  = 5
MODE   = "pqc"
LR     = 0.01

METRICS = ["auc", "sensitivity", "specificity", "f1", "accuracy", "loss"]
TIMING  = ["avg_enc_ms", "dec_ms", "round_ms"]


def run_seed(seed):
    print(f"\n{'═'*64}")
    print(f"  Seed {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})"
          f"  — {ROUNDS} rounds, {NODES} nodes, lr={LR}")
    print(f"{'═'*64}\n")
    subprocess.run(
        [sys.executable, "simulation.py",
         "--rounds", str(ROUNDS),
         "--nodes",  str(NODES),
         "--mode",   MODE,
         "--lr",     str(LR),
         "--seed",   str(seed)],
        check=True,
    )
    # locate the file written by this run
    pattern = f"results_{MODE}_{ROUNDS}r_{NODES}n_seed{seed}_*.json"
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No output file matched {pattern}")
    return matches[-1]


def load_rounds(path, mode=MODE):
    with open(path) as f:
        data = json.load(f)
    return data["rounds_detail"][mode]   # list of 50 dicts


def compute_stats(all_seed_rounds):
    """
    all_seed_rounds: list (one per seed) of list-of-round dicts.
    Returns dict keyed by round number → {metric: {mean, std}}.
    Also returns a flat summary over the final round + timing averages.
    """
    n_rounds = len(all_seed_rounds[0])

    by_round = {}
    for rnd_idx in range(n_rounds):
        rnd_num = rnd_idx + 1
        by_round[rnd_num] = {}
        for m in METRICS + TIMING:
            vals = [seed_rounds[rnd_idx][m] for seed_rounds in all_seed_rounds]
            by_round[rnd_num][m] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
            }

    # final-round quality metrics
    final = {m: by_round[n_rounds][m] for m in METRICS}

    # timing: averaged over all rounds then over seeds
    for t in TIMING:
        per_seed_means = [
            float(np.mean([r[t] for r in seed_rounds]))
            for seed_rounds in all_seed_rounds
        ]
        final[t] = {
            "mean": float(np.mean(per_seed_means)),
            "std":  float(np.std(per_seed_means)),
        }

    return by_round, final


def print_summary(final):
    labels = {
        "auc":         "AUC",
        "sensitivity": "Sensitivity (TPR)",
        "specificity": "Specificity (TNR)",
        "f1":          "F1 Score",
        "accuracy":    "Accuracy",
        "loss":        "Loss",
        "avg_enc_ms":  "Avg enc / node (ms)",
        "dec_ms":      "Avg dec / round (ms)",
        "round_ms":    "Avg round time (ms)",
    }
    fmt = {
        "auc": "{:.4f}", "sensitivity": "{:.4f}", "specificity": "{:.4f}",
        "f1": "{:.4f}", "accuracy": "{:.2%}", "loss": "{:.4f}",
        "avg_enc_ms": "{:.3f}", "dec_ms": "{:.3f}", "round_ms": "{:.1f}",
    }
    print(f"\n{'═'*64}")
    print(f"  MULTI-SEED SUMMARY  "
          f"[{MODE.upper()}  {ROUNDS}r  {NODES}n  lr={LR}  n_seeds={len(SEEDS)}]")
    print(f"{'═'*64}")
    print(f"  {'Metric':<28}  {'Mean':>10}  {'±Std':>10}")
    print(f"  {'─'*28}  {'─'*10}  {'─'*10}")
    for key, label in labels.items():
        s = final[key]
        mean_s = fmt[key].format(s["mean"])
        std_s  = fmt[key].format(s["std"])
        print(f"  {label:<28}  {mean_s:>10}  {std_s:>10}")
    print(f"{'═'*64}\n")


def main():
    all_paths = []
    all_seed_rounds = []

    for seed in SEEDS:
        path = run_seed(seed)
        all_paths.append(path)
        all_seed_rounds.append(load_rounds(path))
        print(f"  → loaded {path}")

    by_round, final = compute_stats(all_seed_rounds)
    print_summary(final)

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "seeds": SEEDS, "rounds": ROUNDS,
            "nodes": NODES, "mode": MODE, "lr": LR,
        },
        "final_summary": final,
        "by_round": by_round,
        "seed_files": all_paths,
    }

    out_path = "results_multiseed.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}\n")


if __name__ == "__main__":
    main()
