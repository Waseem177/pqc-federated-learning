#!/usr/bin/env python3
"""
PQC-Secured Federated Learning Simulation
Usage: python simulation.py [--rounds N] [--nodes N] [--mode pqc|classical|both] [--lr F]
"""

import argparse
import json
import time
from datetime import datetime

import numpy as np

from data_utils import load_and_partition
from node import FederatedNode, FeedForwardNN
from server import AggregationServer


def parse_args():
    p = argparse.ArgumentParser(description="PQC Federated Learning Simulation")
    p.add_argument("--rounds",  type=int,   default=5)
    p.add_argument("--nodes",   type=int,   default=5)
    p.add_argument("--mode",    choices=["pqc", "classical", "both"], default="pqc")
    p.add_argument("--lr",      type=float, default=1e-2)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--dataset", choices=["mixed", "pima"], default="mixed")
    return p.parse_args()


def _bar(title, width=64):
    pad = max(0, (width - len(title) - 2) // 2)
    right = width - pad - len(title) - 2
    print(f"\n{'═'*width}")
    print(f"{'═'*pad} {title} {'═'*right}")
    print(f"{'═'*width}")


def run_one(mode, shards, n_rounds, lr, seed=42):
    n_nodes = len(shards)

    # seed numpy global RNG so training shuffles are reproducible per seed
    np.random.seed(seed)

    # global test set = union of all hospital test sets
    X_te_global = np.concatenate([s[2] for s in shards])
    y_te_global = np.concatenate([s[3] for s in shards])

    print(f"\n[{mode.upper()}] Initialising server...")
    srv = AggregationServer(mode=mode)
    ref_model = FeedForwardNN(input_dim=shards[0][0].shape[1], seed=seed)
    srv.init_weights(ref_model.get_weights())

    print(f"[{mode.upper()}] Creating {n_nodes} nodes  (lr={lr}, seed={seed})...")
    nodes = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(shards):
        nd = FederatedNode(
            node_id=i,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            server_kem_pub=srv.kem_public_key,
            server_rsa_pub=srv.rsa_public_key,
            hmac_key=srv.hmac_key,
            lr=lr,
            mode=mode,
            model_seed=i * 1000 + seed,
        )
        srv.register_node(i, len(X_tr))
        nodes.append(nd)

    print(f"\n{'─'*64}")
    round_log = []

    for rnd in range(1, n_rounds + 1):
        print(f"\n[{mode.upper()} | Round {rnd}/{n_rounds}]")
        t0 = time.perf_counter()

        global_w = srv.global_weights
        prev_w   = [w.copy() for w in global_w]

        bundles, enc_ms_list, overhead_list = [], [], []

        for nd in nodes:
            bundle, timing, metrics, _ = nd.prepare_update(global_w)
            bundles.append(bundle)
            enc_ms_list.append(timing.total_enc_ms())
            overhead_list.append(bundle.overhead_bytes())
            print(
                f"  Node {nd.node_id}: "
                f"auc={metrics['auc']:.3f}  "
                f"sens={metrics['sensitivity']:.3f}  "
                f"spec={metrics['specificity']:.3f}  "
                f"f1={metrics['f1']:.3f}  "
                f"enc={timing.total_enc_ms():.2f} ms"
            )

        agg      = srv.aggregate(bundles)
        round_ms = (time.perf_counter() - t0) * 1000.0
        dec_ms   = sum(t.total_dec_ms() for t in agg["timings"])
        avg_enc  = float(np.mean(enc_ms_list))
        frob     = srv.frob(prev_w)
        net_kb   = agg["total_bytes"] / 1024.0
        overhead = sum(overhead_list)

        gm = FeedForwardNN(input_dim=X_te_global.shape[1])
        gm.set_weights(srv.global_weights)
        gmet = gm.evaluate(X_te_global, y_te_global)

        print(f"\n  ── Round {rnd} {'─'*44}")
        print(
            f"  Global:  auc={gmet['auc']:.3f}  "
            f"sens={gmet['sensitivity']:.3f}  "
            f"spec={gmet['specificity']:.3f}  "
            f"f1={gmet['f1']:.3f}  "
            f"acc={gmet['accuracy']*100:.1f}%  "
            f"loss={gmet['loss']:.4f}"
        )
        print(
            f"  Accepted {len(agg['accepted'])}/{n_nodes}"
            + (f"  | REJECTED {agg['rejected']}" if agg["rejected"] else "")
        )
        print(
            f"  Δ‖W‖={frob:.5f}  |  "
            f"avg enc={avg_enc:.2f} ms  dec={dec_ms:.2f} ms  round={round_ms:.0f} ms"
        )
        print(f"  Network: {net_kb:.1f} KB  |  crypto overhead: {overhead} B")

        round_log.append(dict(
            round=rnd,
            loss=gmet["loss"],         accuracy=gmet["accuracy"],
            f1=gmet["f1"],             auc=gmet["auc"],
            sensitivity=gmet["sensitivity"], specificity=gmet["specificity"],
            avg_enc_ms=avg_enc,        dec_ms=dec_ms,
            round_ms=round_ms,         net_kb=net_kb,
            overhead_b=overhead,       frob=frob,
            accepted=len(agg["accepted"]), rejected=len(agg["rejected"]),
        ))

    return round_log


def summarise(mode, log):
    return dict(
        mode=mode,
        final_auc        =log[-1]["auc"],
        final_sensitivity=log[-1]["sensitivity"],
        final_specificity=log[-1]["specificity"],
        final_f1         =log[-1]["f1"],
        final_accuracy   =log[-1]["accuracy"],
        final_loss       =log[-1]["loss"],
        avg_round_ms =float(np.mean([r["round_ms"]   for r in log])),
        avg_enc_ms   =float(np.mean([r["avg_enc_ms"] for r in log])),
        avg_dec_ms   =float(np.mean([r["dec_ms"]     for r in log])),
        total_net_kb =float(sum(r["net_kb"]           for r in log)),
        overhead_b   =int(log[-1]["overhead_b"]),
        rounds       =len(log),
    )


def print_comparison_table(summaries):
    _bar("FINAL COMPARISON TABLE")
    labels = [
        ("final_auc",         "AUC",                     lambda v: f"{v:.3f}"),
        ("final_sensitivity", "Sensitivity (TPR)",        lambda v: f"{v:.3f}"),
        ("final_specificity", "Specificity (TNR)",        lambda v: f"{v:.3f}"),
        ("final_f1",          "F1 Score",                 lambda v: f"{v:.3f}"),
        ("final_accuracy",    "Accuracy",                 lambda v: f"{v*100:.1f}%"),
        ("final_loss",        "Loss",                     lambda v: f"{v:.4f}"),
        ("avg_round_ms",      "Avg round (ms)",           lambda v: f"{v:.0f}"),
        ("avg_enc_ms",        "Avg enc/node (ms)",        lambda v: f"{v:.2f}"),
        ("avg_dec_ms",        "Avg dec/round (ms)",       lambda v: f"{v:.2f}"),
        ("total_net_kb",      "Total network",            lambda v: f"{v:.1f} KB"),
        ("overhead_b",        "Crypto overhead/round",    lambda v: f"{v} B"),
    ]
    col_w = 26
    modes = [s["mode"].upper() for s in summaries]
    print(f"  {'Metric':<{col_w}}" + "".join(f"  {m:>14}" for m in modes))
    print(f"  {'─'*col_w}" + ("  " + "─"*14) * len(modes))
    for key, label, fmt in labels:
        row = f"  {label:<{col_w}}"
        for s in summaries:
            row += f"  {fmt(s[key]):>14}"
        print(row)
    print()


def main():
    args = parse_args()
    n_rounds, n_nodes, lr = args.rounds, args.nodes, args.lr
    modes = ["pqc", "classical"] if args.mode == "both" else [args.mode]

    _bar(f"PQC-FL Healthcare  |  {args.mode.upper()}  |  {n_nodes} nodes  |  {n_rounds}r  |  lr={lr}")

    print(f"\n[Setup] Loading dataset (dataset={args.dataset})...")
    shards = load_and_partition(n_nodes=n_nodes, verbose=True, dataset=args.dataset)

    all_results = {}
    for mode in modes:
        round_log = run_one(mode, shards, n_rounds, lr, seed=args.seed)
        all_results[mode] = round_log

    summaries = [summarise(m, all_results[m]) for m in modes]

    if len(summaries) == 1:
        s = summaries[0]
        _bar(f"FINAL SUMMARY  [{s['mode'].upper()}]")
        print(f"  AUC              : {s['final_auc']:.3f}")
        print(f"  Sensitivity      : {s['final_sensitivity']:.3f}")
        print(f"  Specificity      : {s['final_specificity']:.3f}")
        print(f"  F1 Score         : {s['final_f1']:.3f}")
        print(f"  Accuracy         : {s['final_accuracy']*100:.1f}%")
        print(f"  Loss             : {s['final_loss']:.4f}")
        print(f"  Avg round time   : {s['avg_round_ms']:.0f} ms")
        print(f"  Avg enc / node   : {s['avg_enc_ms']:.2f} ms")
        print(f"  Avg dec / round  : {s['avg_dec_ms']:.2f} ms")
        print(f"  Total network    : {s['total_net_kb']:.1f} KB  ({n_rounds}r × {n_nodes}n)")
        print(f"  Crypto overhead  : {s['overhead_b']} B/round")
        print(f"{'═'*64}\n")
    else:
        print_comparison_table(summaries)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_{args.mode}_{n_rounds}r_{n_nodes}n_seed{args.seed}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(dict(
            timestamp=ts, mode=args.mode,
            rounds=n_rounds, nodes=n_nodes, lr=lr, seed=args.seed,
            summaries=summaries,
            rounds_detail={m: all_results[m] for m in modes},
        ), f, indent=2)
    print(f"Results saved to {out_path}\n")


if __name__ == "__main__":
    main()
