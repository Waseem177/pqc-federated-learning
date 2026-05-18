#!/usr/bin/env python3
"""
Byzantine Attack Demonstration — PQC-FL Healthcare

Part A — Tampered Signature:
  Rogue node corrupts signature bytes before sending.
  ML-DSA-65 verification fails → node rejected every round.
  Detection rate should be 100%.

Part B — Gradient Poisoning:
  Rogue node scales delta by poison_scale (default 50×) but signs correctly.
  Server-side norm anomaly detection (reject if norm > 3×median) catches it.
  Runs with and without detection to show AUC degradation when undefended.

Usage:
  python byzantine_attack.py [--rounds N] [--attack signature|poisoning|both]
                              [--rogue-node N] [--poison-scale F] [--seed N]
"""

import argparse
import json
from datetime import datetime

import numpy as np

from data_utils import load_and_partition
from node import FederatedNode, FeedForwardNN, weights_to_bytes
from server import AggregationServer
from pqc_layer import encrypt_and_sign


# ── Rogue node variants ────────────────────────────────────────────────────────

class TamperedSigNode(FederatedNode):
    """Part A: valid encrypted bundle, corrupted signature — ML-DSA rejects it."""

    def prepare_update(self, global_weights):
        bundle, timing, metrics, sz = super().prepare_update(global_weights)
        bad_sig = bytearray(bundle.signature)
        for i in range(min(32, len(bad_sig))):
            bad_sig[i] ^= 0xFF
        bundle.signature = bytes(bad_sig)
        return bundle, timing, metrics, sz


class PoisoningNode(FederatedNode):
    """Part B: correctly signed bundle carrying a gradient scaled by poison_scale."""

    def __init__(self, *args, poison_scale=50.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.poison_scale = poison_scale

    def prepare_update(self, global_weights):
        pre = [w.copy() for w in global_weights]
        self.set_global_weights(pre)
        new_w = self.train_local()
        delta = [(n - p) * self.poison_scale for n, p in zip(new_w, pre)]
        delta_bytes = weights_to_bytes(delta)
        metrics = self.model.evaluate(self.X_test, self.y_test)
        bundle, timing = encrypt_and_sign(
            self.node_id, delta_bytes, self.server_kem_pub, self.sig_kp
        )
        return bundle, timing, metrics, len(delta_bytes)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_nodes(shards, srv, rogue_id, rogue_cls, rogue_kwargs, seed):
    nodes = []
    for i, (X_tr, y_tr, X_te, y_te) in enumerate(shards):
        cls = rogue_cls if i == rogue_id else FederatedNode
        kw  = rogue_kwargs if i == rogue_id else {}
        nd  = cls(
            node_id=i,
            X_train=X_tr, y_train=y_tr,
            X_test=X_te,  y_test=y_te,
            server_kem_pub=srv.kem_public_key,
            server_rsa_pub=None,
            hmac_key=b"",
            mode="pqc",
            model_seed=i * 1000 + seed,
            **kw,
        )
        srv.register_node(i, len(X_tr))
        nodes.append(nd)
    return nodes


def _global_eval(shards, srv):
    X_te = np.concatenate([s[2] for s in shards])
    y_te = np.concatenate([s[3] for s in shards])
    gm = FeedForwardNN(input_dim=X_te.shape[1])
    gm.set_weights(srv.global_weights)
    return gm.evaluate(X_te, y_te)


# ── Experiment: Part A ─────────────────────────────────────────────────────────

def run_signature_attack(shards, n_rounds, rogue_id, seed):
    print(f"\n{'═'*64}")
    print(f"  Part A — Tampered Signature  (rogue=node {rogue_id})")
    print(f"{'═'*64}")

    np.random.seed(seed)
    srv = AggregationServer(mode="pqc")
    srv.init_weights(FeedForwardNN(input_dim=shards[0][0].shape[1], seed=seed).get_weights())
    nodes = _build_nodes(shards, srv, rogue_id, TamperedSigNode, {}, seed)

    round_log = []
    for rnd in range(1, n_rounds + 1):
        bundles = [nd.prepare_update(srv.global_weights)[0] for nd in nodes]
        agg = srv.aggregate(bundles)
        detected = rogue_id in agg["rejected"]
        gmet = _global_eval(shards, srv)
        print(
            f"  Round {rnd:2d}: detected={str(detected):<5}  "
            f"accepted={agg['accepted']}  auc={gmet['auc']:.3f}"
        )
        round_log.append({
            "round": rnd, "detected": detected,
            "accepted": agg["accepted"], "rejected": agg["rejected"],
            "auc": gmet["auc"], "accuracy": gmet["accuracy"],
        })

    rate = sum(r["detected"] for r in round_log) / n_rounds
    print(f"\n  Detection rate: {rate*100:.1f}%  ({int(rate*n_rounds)}/{n_rounds} rounds)")
    return round_log, rate


# ── Experiment: Part B ─────────────────────────────────────────────────────────

def run_poisoning_attack(shards, n_rounds, rogue_id, poison_scale, seed):
    print(f"\n{'═'*64}")
    print(f"  Part B — Gradient Poisoning  (rogue=node {rogue_id}, scale={poison_scale}×)")
    print(f"{'═'*64}")

    results = {}
    for norm_filter in [True, False]:
        label = "with_detection" if norm_filter else "without_detection"
        print(f"\n  ── norm_filter={norm_filter} ({label}) ──")
        np.random.seed(seed)

        srv = AggregationServer(mode="pqc")
        srv.init_weights(FeedForwardNN(input_dim=shards[0][0].shape[1], seed=seed).get_weights())
        nodes = _build_nodes(
            shards, srv, rogue_id, PoisoningNode, {"poison_scale": poison_scale}, seed
        )

        round_log = []
        for rnd in range(1, n_rounds + 1):
            bundles = [nd.prepare_update(srv.global_weights)[0] for nd in nodes]
            agg = srv.aggregate(bundles, norm_filter=norm_filter)
            detected = rogue_id in agg["rejected"]
            gmet = _global_eval(shards, srv)

            norms_str = ""
            if agg["norms"]:
                vals = [f"{agg['norms'].get(i, 0.0):.1f}" for i in range(len(shards))]
                norms_str = f"  norms=[{', '.join(vals)}]"

            print(
                f"  Round {rnd:2d}: detected={str(detected):<5}"
                + norms_str +
                f"  auc={gmet['auc']:.3f}"
            )
            round_log.append({
                "round": rnd, "detected": detected,
                "accepted": agg["accepted"], "rejected": agg["rejected"],
                "norms": {str(k): v for k, v in agg["norms"].items()},
                "auc": gmet["auc"], "accuracy": gmet["accuracy"],
            })

        rate = sum(r["detected"] for r in round_log) / n_rounds
        final_auc = round_log[-1]["auc"]
        print(f"  Detection rate: {rate*100:.1f}%  Final AUC: {final_auc:.3f}")
        results[label] = {"rounds": round_log, "detection_rate": rate, "final_auc": final_auc}

    auc_defended   = results["with_detection"]["final_auc"]
    auc_undefended = results["without_detection"]["final_auc"]
    print(f"\n  AUC defended={auc_defended:.3f}  vs  undefended={auc_undefended:.3f}  "
          f"(delta={auc_defended - auc_undefended:+.3f})")
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rounds",       type=int,   default=20)
    p.add_argument("--attack",       choices=["signature", "poisoning", "both"], default="both")
    p.add_argument("--rogue-node",   type=int,   default=4)
    p.add_argument("--poison-scale", type=float, default=50.0)
    p.add_argument("--seed",         type=int,   default=42)
    args = p.parse_args()

    print(f"\n[Setup] Loading dataset (5 nodes)...")
    shards = load_and_partition(n_nodes=5, verbose=False)

    output = {
        "timestamp": datetime.now().isoformat(),
        "rounds": args.rounds,
        "rogue_node": args.rogue_node,
        "seed": args.seed,
    }

    if args.attack in ("signature", "both"):
        log, rate = run_signature_attack(shards, args.rounds, args.rogue_node, args.seed)
        output["part_a"] = {"rounds": log, "detection_rate": rate}

    if args.attack in ("poisoning", "both"):
        res = run_poisoning_attack(
            shards, args.rounds, args.rogue_node, args.poison_scale, args.seed
        )
        output["part_b"] = {"poison_scale": args.poison_scale, **res}

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_byzantine_{args.attack}_{args.rounds}r_seed{args.seed}_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
