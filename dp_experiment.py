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
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()

    print(f"\nLoading dataset (5 nodes)...")
    shards = load_and_partition(n_nodes=5, verbose=False)
    print(f"Epsilons: {EPSILONS}")
    print(f"Rounds: {args.rounds}  |  Seed: {args.seed}")
    print(f"DP params: clip={DP_CLIP}, delta={DP_DELTA}")

    results = []
    for eps in EPSILONS:
        res = run_one(shards, args.rounds, eps, args.seed)
        results.append(res)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": datetime.now().isoformat(),
        "rounds": args.rounds,
        "seed": args.seed,
        "dp_clip": DP_CLIP,
        "dp_delta": DP_DELTA,
        "results": results,
    }
    path = f"results_dp_sweep_{args.rounds}r_seed{args.seed}_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {path}")

    # Summary table
    print(f"\n{'Epsilon':>12}  {'Final AUC':>10}  {'sigma':>10}")
    print("─" * 38)
    for r in results:
        eps = r["epsilon"]
        auc = r["final_auc"]
        if eps == "inf":
            sigma_str = "0 (no DP)"
        else:
            sigma_str = f"{_dp_noise_sigma(float(eps)):.4f}"
        print(f"{eps:>12}  {auc:>10.3f}  {sigma_str:>10}")


if __name__ == "__main__":
    main()
