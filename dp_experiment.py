#!/usr/bin/env python3
"""
Differential Privacy epsilon sweep — PQC-FL Healthcare

Runs federated training at each epsilon value (and ε=∞ as baseline).
Saves per-round AUC for all epsilons to JSON, ready for fig5.

Usage:
  python dp_experiment.py [--rounds N] [--seed N]
"""

import argparse
import json
from datetime import datetime

import numpy as np

from data_utils import load_and_partition
from node import FederatedNode, FeedForwardNN, DP_CLIP, DP_DELTA, _dp_noise_sigma
from server import AggregationServer

EPSILONS = [0.5, 1.0, 5.0, 10.0, 50.0, float("inf")]


def run_one(shards, n_rounds, epsilon, seed):
    label = f"ε={epsilon}" if epsilon != float("inf") else "ε=∞ (no DP)"
    eps_str = str(epsilon) if epsilon != float("inf") else "inf"
    print(f"\n{'─'*56}")
    print(f"  {label}  (seed={seed})")
    if epsilon != float("inf"):
        sigma = _dp_noise_sigma(epsilon)
        print(f"  σ = {sigma:.4f}  (clip={DP_CLIP}, δ={DP_DELTA})")
    print(f"{'─'*56}")

    np.random.seed(seed)
    srv = AggregationServer(mode="pqc")
    srv.init_weights(FeedForwardNN(input_dim=shards[0][0].shape[1], seed=seed).get_weights())

    nodes = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(shards):
        eps_arg = None if epsilon == float("inf") else epsilon
        nd = FederatedNode(
            node_id=i,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            server_kem_pub=srv.kem_public_key,
            server_rsa_pub=None, hmac_key=b"",
            mode="pqc",
            lr=0.01,
            model_seed=i * 1000 + seed,
            epsilon=eps_arg,
        )
        srv.register_node(i, len(X_tr))
        nodes.append(nd)

    round_log = []
    X_te_all = np.concatenate([s[2] for s in shards])
    y_te_all = np.concatenate([s[3] for s in shards])
    gm = FeedForwardNN(input_dim=X_te_all.shape[1])

    for rnd in range(1, n_rounds + 1):
        bundles = [nd.prepare_update(srv.global_weights)[0] for nd in nodes]
        srv.aggregate(bundles)
        gm.set_weights(srv.global_weights)
        met = gm.evaluate(X_te_all, y_te_all)
        print(f"  Round {rnd:2d}:  auc={met['auc']:.3f}  acc={met['accuracy']:.3f}")
        round_log.append({"round": rnd, **met})

    final_auc = round_log[-1]["auc"]
    print(f"  Final AUC: {final_auc:.3f}")
    return {"epsilon": eps_str, "rounds": round_log, "final_auc": final_auc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--seeds",  type=int, nargs="+", default=[42, 123, 456])
    args = p.parse_args()

    print(f"\nLoading dataset (5 nodes, pima)...")
    shards, _ = load_and_partition(n_nodes=5, verbose=False, dataset="pima")
    print(f"Epsilons: {EPSILONS}")
    print(f"Rounds: {args.rounds}  |  Seeds: {args.seeds}")
    print(f"DP params: clip={DP_CLIP}, delta={DP_DELTA}")

    # per_eps[eps_str] = list of final_aucs across seeds
    per_eps = {(str(e) if e != float("inf") else "inf"): [] for e in EPSILONS}
    seed_results = []

    for seed in args.seeds:
        print(f"\n{'━'*56}  seed={seed}")
        seed_run = []
        for eps in EPSILONS:
            res = run_one(shards, args.rounds, eps, seed)
            seed_run.append(res)
            per_eps[res["epsilon"]].append(res["final_auc"])
        seed_results.append({"seed": seed, "results": seed_run})

    # Summary table
    print(f"\n{'Epsilon':>12}  {'AUC mean':>10}  {'AUC std':>9}  {'sigma':>10}")
    print("─" * 48)
    summary = {}
    for eps in EPSILONS:
        eps_str = "inf" if eps == float("inf") else str(eps)
        aucs = per_eps[eps_str]
        mean, std = float(np.mean(aucs)), float(np.std(aucs))
        sigma_str = "0 (no DP)" if eps == float("inf") else f"{_dp_noise_sigma(float(eps)):.4f}"
        print(f"{eps_str:>12}  {mean:>10.3f}  {std:>9.3f}  {sigma_str:>10}")
        summary[eps_str] = {"mean": mean, "std": std, "aucs": aucs}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    n_seeds = len(args.seeds)
    path = f"results_dp_sweep_{args.rounds}r_{n_seeds}seeds_{ts}.json"
    out = {
        "timestamp": datetime.now().isoformat(),
        "rounds": args.rounds,
        "seeds": args.seeds,
        "dp_clip": DP_CLIP,
        "dp_delta": DP_DELTA,
        "seed_results": seed_results,
        "summary": summary,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()


# ── RDP Composition Accounting ─────────────────────────────────────────────
# Mironov (2017) RDP for Gaussian mechanism, composed over T rounds,
# converted to (ε, δ)-DP via Balle et al. (2020).

def compute_rdp_gaussian(sigma, sensitivity, alpha):
    """RDP ε for Gaussian mechanism at order alpha (Mironov 2017, Prop. 3)."""
    noise_multiplier = sigma / sensitivity
    return alpha / (2 * noise_multiplier ** 2)


def rdp_to_dp(rdp_eps, alpha, delta):
    """Convert (alpha, rdp_eps)-RDP to (ε, δ)-DP (Balle et al. 2020, Prop. 3)."""
    if alpha <= 1:
        return np.inf
    return rdp_eps + np.log((alpha - 1) / alpha) \
           - (np.log(delta) + np.log(alpha - 1)) / (alpha - 1)


def compute_epsilon(num_rounds, sigma, sensitivity=0.1, delta=1e-5, alphas=None):
    """Compose RDP over num_rounds rounds; return tightest (ε, δ)-DP bound."""
    if alphas is None:
        alphas = sorted(set(list(range(2, 512)) + [a / 10 for a in range(11, 200)]))
    best_eps = np.inf
    for alpha in alphas:
        rdp_composed = num_rounds * compute_rdp_gaussian(sigma, sensitivity, alpha)
        eps_dp = rdp_to_dp(rdp_composed, alpha, delta)
        if eps_dp < best_eps:
            best_eps = eps_dp
    return best_eps


def print_rdp_tables():
    """Print cumulative privacy loss tables for paper (Table V)."""
    SENSITIVITY = 0.1
    DELTA = 1e-5

    # σ = S * √(2 ln(1.25/δ)) / ε  (single-round Gaussian calibration)
    configs = [
        ("ε=0.5",  0.5,  0.969),
        ("ε=1.0",  1.0,  0.484),
        ("ε=5.0",  5.0,  0.097),
        ("ε=10.0", 10.0, 0.048),
        ("ε=50.0", 50.0, 0.010),
    ]
    checkpoints = [1, 10, 30, 50]

    print("\n" + "="*70)
    print("TABLE V — CUMULATIVE PRIVACY LOSS UNDER RDP COMPOSITION")
    print(f"  δ = {DELTA}, sensitivity S = {SENSITIVITY}")
    print("="*70)
    print(f"{'Config':<10} {'σ':<8} " +
          "  ".join(f"T={T:>3} ε_cumul" for T in checkpoints))
    print("-"*70)
    for label, _, sigma in configs:
        row = f"{label:<10} {sigma:<8.3f} "
        for T in checkpoints:
            row += f"  {compute_epsilon(T, sigma, SENSITIVITY, DELTA):>12.3f}"
        print(row)
    print("="*70)
    print("\nNote: Cumulative ε grows as O(√T) under RDP, tighter than naive O(T).")

    print("\n── For paper (ε=10 config, σ=0.048): ──")
    sigma_main = 0.048
    print(f"{'Rounds':<10} {'ε_naive':>10} {'ε_RDP (ours)':>14} {'Saving':>10}")
    print("-"*46)
    for T in checkpoints:
        eps_rdp = compute_epsilon(T, sigma_main, SENSITIVITY, DELTA)
        saving = 10.0 * T - eps_rdp
        print(f"{T:<10} {10.0*T:>10.1f} {eps_rdp:>14.3f} {saving:>10.3f}")


if __name__ == "__main__":
    print_rdp_tables()
