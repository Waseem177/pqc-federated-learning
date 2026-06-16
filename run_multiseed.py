#!/usr/bin/env python3
"""
Multi-seed evaluation harness.
Runs simulation.py 5 times with different seeds, aggregates mean ± std,
and saves results_multiseed.json.
"""

import argparse
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

METRICS     = ["auc", "sensitivity", "specificity", "f1", "accuracy", "loss"]
DS_METRICS  = ["diabetes_auc", "heart_auc", "cancer_auc"]   # hetero mode only
TIMING      = ["avg_enc_ms", "dec_ms", "round_ms"]


def run_seed(seed, dataset="hetero"):
    print(f"\n{'═'*64}")
    print(f"  Seed {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})"
          f"  — {ROUNDS} rounds, {NODES} nodes, lr={LR}, dataset={dataset}")
    print(f"{'═'*64}\n")
    subprocess.run(
        [sys.executable, "simulation.py",
         "--rounds",  str(ROUNDS),
         "--nodes",   str(NODES),
         "--mode",    MODE,
         "--lr",      str(LR),
         "--seed",    str(seed),
         "--dataset", dataset],
        check=True,
    )
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
    # determine which per-dataset metrics are present in these results
    active_ds_metrics = [m for m in DS_METRICS
                         if m in all_seed_rounds[0][0]]

    for rnd_idx in range(n_rounds):
        rnd_num = rnd_idx + 1
        by_round[rnd_num] = {}
        for m in METRICS + TIMING + active_ds_metrics:
            vals = [seed_rounds[rnd_idx][m] for seed_rounds in all_seed_rounds]
            by_round[rnd_num][m] = {
                "mean": float(np.mean(vals)),
                "std":  float(np.std(vals)),
            }

    # final-round quality metrics (global + per-dataset if available)
    final = {m: by_round[n_rounds][m] for m in METRICS + active_ds_metrics}

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
        "auc":           "AUC (global)",
        "diabetes_auc":  "AUC — Diabetes",
        "heart_auc":     "AUC — Heart Disease",
        "cancer_auc":    "AUC — Breast Cancer",
        "sensitivity":   "Sensitivity (TPR)",
        "specificity":   "Specificity (TNR)",
        "f1":            "F1 Score",
        "accuracy":      "Accuracy",
        "loss":          "Loss",
        "avg_enc_ms":    "Avg enc / node (ms)",
        "dec_ms":        "Avg dec / round (ms)",
        "round_ms":      "Avg round time (ms)",
    }
    fmt = {
        "auc": "{:.4f}", "diabetes_auc": "{:.4f}", "heart_auc": "{:.4f}",
        "cancer_auc": "{:.4f}", "sensitivity": "{:.4f}", "specificity": "{:.4f}",
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
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["hetero", "mixed", "pima"], default="hetero")
    args = p.parse_args()

    all_paths = []
    all_seed_rounds = []

    for seed in SEEDS:
        path = run_seed(seed, dataset=args.dataset)
        all_paths.append(path)
        all_seed_rounds.append(load_rounds(path))
        print(f"  → loaded {path}")

    by_round, final = compute_stats(all_seed_rounds)
    print_summary(final)

    suffix = f"_{args.dataset}" if args.dataset != "hetero" else ""
    out_path = f"results_multiseed{suffix}.json"

    output = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "config": {
            "seeds": SEEDS, "rounds": ROUNDS,
            "nodes": NODES, "mode": MODE, "lr": LR,
            "dataset": args.dataset,
        },
        "final_summary": final,
        "by_round": by_round,
        "seed_files": all_paths,
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {out_path}\n")


if __name__ == "__main__":
    main()
