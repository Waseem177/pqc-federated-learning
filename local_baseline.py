#!/usr/bin/env python3
"""
Local (no-federation) baseline for the heterogeneous 5-hospital setup.

Each hospital trains in complete isolation for N_ROUNDS × LOCAL_EPOCHS epochs
(same total gradient steps as the federated run) and evaluates on its own
test set.  Results are compared against the federated AUC stored in
results_multiseed.json.

Usage: python local_baseline.py
"""

import json
import numpy as np
from data_utils import load_and_partition
from node import FeedForwardNN

SEEDS        = [42, 123, 456, 789, 1337]
N_ROUNDS     = 50
LOCAL_EPOCHS = 5
LR           = 0.01
BATCH_SIZE   = 32

# Dataset label → disease group (matches simulation.py grouping)
_DISEASE_GROUP = {
    "diabetes":       "diabetes",
    "diabetes_noisy": "diabetes",
    "heart":          "heart",
    "cancer":         "cancer",
}


def train_local(X_tr, y_tr, X_te, y_te, seed, n_rounds=N_ROUNDS,
                local_epochs=LOCAL_EPOCHS, lr=LR, batch_size=BATCH_SIZE):
    """Train a node-local model for n_rounds × local_epochs epochs."""
    np.random.seed(seed)
    model = FeedForwardNN(input_dim=X_tr.shape[1], seed=seed)
    n = len(X_tr)
    for _ in range(n_rounds * local_epochs):
        idx = np.random.permutation(n)
        for s in range(0, n, batch_size):
            b = idx[s:s + batch_size]
            model.forward(X_tr[b])
            model.backward(y_tr[b], lr=lr)
    return model.evaluate(X_te, y_te)["auc"]


def run(seed):
    shards, labels = load_and_partition(n_nodes=5, seed=seed,
                                        dataset="hetero", verbose=False)
    node_names = ["Pima-A", "Pima-B", "Cleveland", "WDBC", "Pima-C(noisy)"]
    results = {}
    for i, ((X_tr, y_tr, X_te, y_te), lab) in enumerate(zip(shards, labels)):
        auc = train_local(X_tr, y_tr, X_te, y_te, seed=seed + i)
        group = _DISEASE_GROUP[lab]
        results.setdefault(group, []).append(auc)
        print(f"  Node {i} [{node_names[i]:16s}] ({lab:16s}) → local AUC = {auc:.4f}")
    return results


def main():
    print(f"\nLocal baseline: {len(SEEDS)} seeds × "
          f"{N_ROUNDS} rounds × {LOCAL_EPOCHS} local epochs (no federation)\n")

    all_seed_results = []
    for seed in SEEDS:
        print(f"── Seed {seed} ──────────────────────────────────────────────")
        all_seed_results.append(run(seed))
    print()

    # Aggregate per disease group over seeds
    groups = ["diabetes", "heart", "cancer"]
    local_stats = {}
    for g in groups:
        vals = []
        for sr in all_seed_results:
            vals.extend(sr.get(g, []))
        local_stats[g] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                          "n_obs": len(vals)}

    # Load federated results for comparison
    fed_stats = {}
    try:
        d = json.load(open("results_multiseed.json"))
        fs = d["final_summary"]
        for g in groups:
            key = f"{g}_auc"
            if key in fs:
                fed_stats[g] = fs[key]
    except FileNotFoundError:
        print("  [results_multiseed.json not found — no federated comparison]")

    # Print comparison table
    print("=" * 68)
    print("LOCAL vs. FEDERATED AUC (Round 50, 5 Seeds)")
    print(f"{'Disease':<14} {'Local AUC':>14} {'Federated AUC':>16} {'Δ AUC':>10}")
    print("-" * 68)
    for g in groups:
        l = local_stats[g]
        local_str = f"{l['mean']:.3f} ± {l['std']:.3f}"
        if g in fed_stats:
            f = fed_stats[g]
            fed_str = f"{f['mean']:.3f} ± {f['std']:.3f}"
            delta = f['mean'] - l['mean']
            delta_str = f"{delta:+.3f}"
        else:
            fed_str = "N/A"
            delta_str = "N/A"
        print(f"{g.capitalize():<14} {local_str:>14} {fed_str:>16} {delta_str:>10}")
    print("=" * 68)
    print("\nPositive Δ AUC = federation helps this disease group.")

    out = {
        "seeds": SEEDS, "n_rounds": N_ROUNDS, "local_epochs": LOCAL_EPOCHS,
        "local_auc": local_stats,
        "federated_auc": fed_stats,
    }
    with open("results_local_baseline.json", "w") as f:
        json.dump(out, f, indent=2)
    print("Saved results_local_baseline.json\n")


if __name__ == "__main__":
    main()
