#!/usr/bin/env python3
"""
Security Level Comparison — ML-KEM-512/768/1024 × ML-DSA-44/65/87

Benchmarks all three NIST post-quantum security levels on a realistic
model delta payload. Reports per-operation latency (mean ± std) and
bundle sizes. Suitable for direct use in paper Table/Figure.

Usage:
  python security_levels.py [--iterations N] [--payload-kb N]
"""

import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
import oqs
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

LEVELS = [
    {"label": "Level 1", "nist_level": 1, "kem": "ML-KEM-512",  "sig": "ML-DSA-44"},
    {"label": "Level 3", "nist_level": 3, "kem": "ML-KEM-768",  "sig": "ML-DSA-65"},
    {"label": "Level 5", "nist_level": 5, "kem": "ML-KEM-1024", "sig": "ML-DSA-87"},
]

AES_NONCE_LEN = 12
AES_KEY_LEN   = 32
HKDF_INFO     = b"pq-fedhealth-v1"


def _hkdf(ss):
    return HKDF(algorithm=hashes.SHA256(), length=AES_KEY_LEN, salt=None, info=HKDF_INFO).derive(ss)


def _ms(t0):
    return (time.perf_counter() - t0) * 1000.0


def benchmark_level(kem_alg, sig_alg, payload, n_iter):
    keys = ["keygen_kem_ms", "keygen_sig_ms",
            "encap_ms", "aes_enc_ms", "sign_ms",
            "decap_ms", "aes_dec_ms", "verify_ms",
            "total_enc_ms", "total_dec_ms", "round_trip_ms"]
    raw = {k: [] for k in keys}
    sizes = {}

    for i in range(n_iter):
        # ── Key generation ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        with oqs.KeyEncapsulation(kem_alg) as kem:
            server_pub = kem.generate_keypair()
            server_sec = kem.export_secret_key()
        raw["keygen_kem_ms"].append(_ms(t0))

        t0 = time.perf_counter()
        with oqs.Signature(sig_alg) as sig:
            node_pub = sig.generate_keypair()
            node_sec = sig.export_secret_key()
        raw["keygen_sig_ms"].append(_ms(t0))

        # ── Encrypt + sign ────────────────────────────────────────────────────
        t0 = time.perf_counter()
        with oqs.KeyEncapsulation(kem_alg) as kem:
            kem_ct, ss = kem.encap_secret(server_pub)
        raw["encap_ms"].append(_ms(t0))

        nonce = os.urandom(AES_NONCE_LEN)
        t0 = time.perf_counter()
        aes_ct = AESGCM(_hkdf(ss)).encrypt(nonce, payload, None)
        raw["aes_enc_ms"].append(_ms(t0))

        msg = kem_ct + aes_ct
        t0 = time.perf_counter()
        with oqs.Signature(sig_alg, secret_key=node_sec) as sig:
            signature = sig.sign(msg)
        raw["sign_ms"].append(_ms(t0))

        enc_ms = raw["encap_ms"][-1] + raw["aes_enc_ms"][-1] + raw["sign_ms"][-1]
        raw["total_enc_ms"].append(enc_ms)

        # ── Decrypt + verify ──────────────────────────────────────────────────
        t0 = time.perf_counter()
        with oqs.KeyEncapsulation(kem_alg, secret_key=server_sec) as kem:
            ss2 = kem.decap_secret(kem_ct)
        raw["decap_ms"].append(_ms(t0))

        t0 = time.perf_counter()
        AESGCM(_hkdf(ss2)).decrypt(nonce, aes_ct, None)
        raw["aes_dec_ms"].append(_ms(t0))

        t0 = time.perf_counter()
        with oqs.Signature(sig_alg) as sig:
            valid = sig.verify(msg, signature, node_pub)
        raw["verify_ms"].append(_ms(t0))
        assert valid, f"Signature verification failed for {sig_alg}"

        dec_ms = raw["decap_ms"][-1] + raw["aes_dec_ms"][-1] + raw["verify_ms"][-1]
        raw["dec_ms"] = raw.get("dec_ms", [])
        raw["total_dec_ms"].append(dec_ms)
        raw["round_trip_ms"].append(enc_ms + dec_ms)

        if i == 0:
            sizes = {
                "kem_ct_bytes":          len(kem_ct),
                "aes_ct_bytes":          len(aes_ct),
                "signature_bytes":       len(signature),
                "sig_pubkey_bytes":      len(node_pub),
                "nonce_bytes":           AES_NONCE_LEN,
                "payload_bytes":         len(payload),
                "total_bundle_bytes":    len(kem_ct) + len(aes_ct) + AES_NONCE_LEN + len(signature) + len(node_pub),
                "crypto_overhead_bytes": len(kem_ct) + AES_NONCE_LEN + len(signature) + len(node_pub),
            }

    timings = {
        k: {"mean_ms": float(np.mean(v)), "std_ms": float(np.std(v))}
        for k, v in raw.items() if v
    }
    return {"timings": timings, "sizes": sizes}


def print_table(results):
    col = 14
    header_keys = [
        ("total_enc_ms",  "Enc (ms)"),
        ("total_dec_ms",  "Dec (ms)"),
        ("round_trip_ms", "Round-trip (ms)"),
        ("encap_ms",      "KEM encap (ms)"),
        ("sign_ms",       "Sign (ms)"),
        ("decap_ms",      "KEM decap (ms)"),
        ("verify_ms",     "Verify (ms)"),
    ]
    size_keys = [
        ("kem_ct_bytes",          "KEM ciphertext"),
        ("signature_bytes",       "Signature"),
        ("sig_pubkey_bytes",      "Sig public key"),
        ("crypto_overhead_bytes", "Crypto overhead"),
        ("total_bundle_bytes",    "Total bundle"),
    ]

    labels = [r["label"] for r in results]
    print(f"\n{'─'*72}")
    print(f"  {'Metric':<28}" + "".join(f"  {l:>{col}}" for l in labels))
    print(f"  {'─'*28}" + ("  " + "─"*col) * len(labels))

    for key, display in header_keys:
        row = f"  {display:<28}"
        for r in results:
            v = r["result"]["timings"][key]["mean_ms"]
            row += f"  {v:>{col}.3f}"
        print(row)

    print(f"  {'─'*28}" + ("  " + "─"*col) * len(labels))
    for key, display in size_keys:
        row = f"  {display:<28}"
        for r in results:
            v = r["result"]["sizes"][key]
            row += f"  {v:>{col},}"
        print(row)
    print(f"{'─'*72}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=50,
                   help="Iterations per level (default 50)")
    p.add_argument("--payload-kb", type=float, default=12.0,
                   help="Payload size in KB, representative of a model delta (default 12)")
    args = p.parse_args()

    payload = os.urandom(int(args.payload_kb * 1024))
    print(f"\nSecurity Level Comparison")
    print(f"  Payload: {len(payload)/1024:.1f} KB  |  Iterations: {args.iterations} per level")

    all_results = []
    for lvl in LEVELS:
        print(f"\n  Benchmarking {lvl['label']} ({lvl['kem']} + {lvl['sig']})...", flush=True)
        result = benchmark_level(lvl["kem"], lvl["sig"], payload, args.iterations)
        all_results.append({**lvl, "result": result})
        enc  = result["timings"]["total_enc_ms"]["mean_ms"]
        dec  = result["timings"]["total_dec_ms"]["mean_ms"]
        oh   = result["sizes"]["crypto_overhead_bytes"]
        print(f"    enc={enc:.3f} ms  dec={dec:.3f} ms  overhead={oh:,} B")

    print_table(all_results)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results_security_levels_{args.iterations}iter_{ts}.json"

    serialisable = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "result"}
        entry["result"] = r["result"]
        serialisable.append(entry)

    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "iterations": args.iterations,
            "payload_bytes": len(payload),
            "levels": serialisable,
        }, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
