#!/usr/bin/env python3
"""
Scalability experiment: 5, 10, 20 nodes × 3 seeds × 50 rounds.
Saves results_scalability.json and fig7_scalability.png.
"""

import json
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data_utils import load_and_partition
from node import FederatedNode, FeedForwardNN
from server import AggregationServer

# ── Style (matches figures.py) ────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
})
BLUE  = "#2166AC"
RED   = "#D6604D"
GREEN = "#4DAC26"

NODE_COUNTS = [5, 10, 20]
SEEDS       = [42, 123, 456]
N_ROUNDS    = 50
DATASET     = "mixed"


def run_one(n_nodes, seed, n_rounds=N_ROUNDS):
    shards = load_and_partition(n_nodes=n_nodes, seed=seed,
                                dataset=DATASET, verbose=False)
    np.random.seed(seed)
    X_te = np.concatenate([s[2] for s in shards])
    y_te = np.concatenate([s[3] for s in shards])

    srv = AggregationServer(mode="pqc")
    ref = FeedForwardNN(input_dim=shards[0][0].shape[1], seed=seed)
    srv.init_weights(ref.get_weights())

    nodes = []
    for i, (Xtr, ytr, Xte, yte) in enumerate(shards):
        nd = FederatedNode(
            node_id=i, X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
            server_kem_pub=srv.kem_public_key, server_rsa_pub=None,
            hmac_key=b"", lr=1e-3, mode="pqc", model_seed=i * 1000 + seed,
        )
        srv.register_node(i, len(Xtr))
        nodes.append(nd)

    round_times, aucs = [], []
    for _ in range(n_rounds):
        t0 = time.perf_counter()
        global_w = srv.global_weights
        bundles = [nd.prepare_update(global_w)[0] for nd in nodes]
        srv.aggregate(bundles)
        round_times.append((time.perf_counter() - t0) * 1000.0)

        gm = FeedForwardNN(input_dim=X_te.shape[1])
        gm.set_weights(srv.global_weights)
        aucs.append(gm.evaluate(X_te, y_te)["auc"])

    return {
        "final_auc":      float(aucs[-1]),
        "avg_round_ms":   float(np.mean(round_times)),
        "auc_curve":      [float(a) for a in aucs],
        "round_ms_curve": [float(t) for t in round_times],
    }


def run_experiment():
    results = {}
    total = len(NODE_COUNTS) * len(SEEDS)
    done  = 0
    for n in NODE_COUNTS:
        results[n] = []
        for s in SEEDS:
            done += 1
            print(f"  [{done}/{total}] nodes={n}  seed={s} ...", flush=True)
            r = run_one(n, s)
            r["n_nodes"] = n
            r["seed"]    = s
            results[n].append(r)
            print(f"         AUC={r['final_auc']:.3f}  "
                  f"avg_round={r['avg_round_ms']:.0f} ms")

    out = {
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "dataset": DATASET, "rounds": N_ROUNDS, "seeds": SEEDS,
        "node_counts": NODE_COUNTS,
        "results": {str(n): results[n] for n in NODE_COUNTS},
    }
    with open("results_scalability.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  Saved results_scalability.json")
    return results


def plot_figure(results):
    nodes = NODE_COUNTS
    auc_means  = [np.mean([r["final_auc"]    for r in results[n]]) for n in nodes]
    auc_stds   = [np.std( [r["final_auc"]    for r in results[n]]) for n in nodes]
    time_means = [np.mean([r["avg_round_ms"] for r in results[n]]) for n in nodes]
    time_stds  = [np.std( [r["avg_round_ms"] for r in results[n]]) for n in nodes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: AUC vs node count
    ax1.errorbar(nodes, auc_means, yerr=auc_stds, fmt="o-",
                 color=BLUE, linewidth=2, markersize=7,
                 capsize=5, capthick=1.5, label="Final AUC (mean ± std)")
    ax1.set_xlabel("Number of Hospital Nodes")
    ax1.set_ylabel("AUC (ROC)")
    ax1.set_title("Model Quality vs. Federation Scale")
    ax1.set_xticks(nodes)
    ax1.set_ylim(0.5, 1.0)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Panel 2: round latency vs node count
    ax2.errorbar(nodes, time_means, yerr=time_stds, fmt="s-",
                 color=RED, linewidth=2, markersize=7,
                 capsize=5, capthick=1.5, label="Avg round time (mean ± std)")
    ax2.set_xlabel("Number of Hospital Nodes")
    ax2.set_ylabel("Average Round Time (ms)")
    ax2.set_title("Communication Overhead vs. Federation Scale")
    ax2.set_xticks(nodes)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"PQC-FL Scalability: {N_ROUNDS} rounds, {len(SEEDS)} seeds, "
        f"ML-KEM-768 + ML-DSA-65",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    plt.savefig("fig7_scalability.png")
    print("  Saved fig7_scalability.png")


if __name__ == "__main__":
    print(f"\nScalability experiment: nodes={NODE_COUNTS}, "
          f"seeds={SEEDS}, rounds={N_ROUNDS}, dataset={DATASET}\n")
    results = run_experiment()
    plot_figure(results)
    print("\nDone.")
