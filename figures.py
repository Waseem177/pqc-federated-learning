#!/usr/bin/env python3
"""
Generate all paper figures for PQC-FL Healthcare project.

Figures produced:
  fig1_convergence.png      — FL convergence curve (AUC ± std, 5 seeds, 50 rounds)
  fig2_security_levels.png  — Security level vs latency and overhead (NIST L1/L3/L5)
  fig3_byzantine.png        — Byzantine defense: clean vs undefended vs defended
  fig4_latency_breakdown.png — PQC vs Classical per-round latency breakdown

Usage:
  python figures.py
"""

import json
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "legend.fontsize":  10,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
})

BLUE   = "#2166AC"
RED    = "#D6604D"
GREEN  = "#4DAC26"
ORANGE = "#F4A582"
GRAY   = "#888888"


# ── Figure 1: FL Convergence ───────────────────────────────────────────────────

def fig1_convergence(path="results_multiseed.json"):
    d = json.load(open(path))
    by_round = d["by_round"]
    rounds = sorted(int(r) for r in by_round.keys())

    auc_mean = np.array([by_round[str(r)]["auc"]["mean"]   for r in rounds])
    auc_std  = np.array([by_round[str(r)]["auc"]["std"]    for r in rounds])
    acc_mean = np.array([by_round[str(r)]["accuracy"]["mean"] for r in rounds])
    acc_std  = np.array([by_round[str(r)]["accuracy"]["std"]  for r in rounds])

    fig, ax = plt.subplots(figsize=(6.5, 4))

    ax.plot(rounds, auc_mean, color=BLUE, lw=2, label="AUC (mean, 5 seeds)")
    ax.fill_between(rounds, auc_mean - auc_std, auc_mean + auc_std,
                    color=BLUE, alpha=0.20, label="AUC ± 1 std")

    ax.plot(rounds, acc_mean, color=GREEN, lw=2, ls="--", label="Accuracy (mean)")
    ax.fill_between(rounds, acc_mean - acc_std, acc_mean + acc_std,
                    color=GREEN, alpha=0.15)

    final_auc = auc_mean[-1]
    ax.axhline(final_auc, color=BLUE, lw=0.8, ls=":", alpha=0.7)
    ax.annotate(f"Final AUC = {final_auc:.3f}",
                xy=(rounds[-1], final_auc), xytext=(-5, 6),
                textcoords="offset points", ha="right",
                fontsize=9, color=BLUE)

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Metric Value")
    ax.set_title("PQC-FL Convergence — 5 Hospital Nodes (5 Seeds, 50 Rounds)")
    ax.set_xlim(1, max(rounds))
    ax.set_ylim(0.3, 1.0)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("fig1_convergence.png")
    plt.close(fig)
    print("Saved fig1_convergence.png")


# ── Figure 2: Security Level Comparison ───────────────────────────────────────

def fig2_security_levels(path="results_security_levels_20iter_20260518_063541.json"):
    d = json.load(open(path))
    levels = d["levels"]

    labels   = [l["label"] for l in levels]
    kem_sigs = [f"{l['kem']}\n+{l['sig']}" for l in levels]

    enc_mean = np.array([l["result"]["timings"]["total_enc_ms"]["mean_ms"]   for l in levels])
    enc_std  = np.array([l["result"]["timings"]["total_enc_ms"]["std_ms"]    for l in levels])
    dec_mean = np.array([l["result"]["timings"]["total_dec_ms"]["mean_ms"]   for l in levels])
    dec_std  = np.array([l["result"]["timings"]["total_dec_ms"]["std_ms"]    for l in levels])
    rt_mean  = np.array([l["result"]["timings"]["round_trip_ms"]["mean_ms"]  for l in levels])

    overhead_kb = np.array([l["result"]["sizes"]["crypto_overhead_bytes"] / 1024 for l in levels])
    bundle_kb   = np.array([l["result"]["sizes"]["total_bundle_bytes"]    / 1024 for l in levels])
    sig_kb      = np.array([l["result"]["sizes"]["signature_bytes"]       / 1024 for l in levels])
    kem_ct_kb   = np.array([l["result"]["sizes"]["kem_ct_bytes"]          / 1024 for l in levels])

    x = np.arange(len(labels))
    width = 0.28

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: latency
    bars_enc = ax1.bar(x - width/2, enc_mean, width, label="Encrypt + Sign",
                       color=BLUE, yerr=enc_std, capsize=4, error_kw={"lw": 1.2})
    bars_dec = ax1.bar(x + width/2, dec_mean, width, label="Decrypt + Verify",
                       color=RED, yerr=dec_std, capsize=4, error_kw={"lw": 1.2})

    for bar, val in zip(bars_enc, enc_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)
    for bar, val in zip(bars_dec, dec_mean):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{val:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(kem_sigs, fontsize=9)
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Per-Operation Latency by NIST Security Level")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, max(enc_mean.max(), dec_mean.max()) * 1.35)

    # Right: overhead stacked bar (KEM ciphertext + signature + pubkey)
    pubkey_kb = np.array([l["result"]["sizes"]["sig_pubkey_bytes"] / 1024 for l in levels])
    nonce_kb  = 12 / 1024  # constant

    p1 = ax2.bar(x, kem_ct_kb, width*2.4, label="KEM ciphertext", color=BLUE)
    p2 = ax2.bar(x, sig_kb,    width*2.4, bottom=kem_ct_kb, label="Signature", color=RED)
    p3 = ax2.bar(x, pubkey_kb, width*2.4, bottom=kem_ct_kb + sig_kb, label="Sig public key", color=GREEN)

    total = kem_ct_kb + sig_kb + pubkey_kb
    for i, (xi, tot) in enumerate(zip(x, total)):
        ax2.text(xi, tot + 0.1, f"{tot:.1f} KB", ha="center", va="bottom", fontsize=8.5)

    ax2.set_xticks(x)
    ax2.set_xticklabels(kem_sigs, fontsize=9)
    ax2.set_ylabel("Size (KB)")
    ax2.set_title("Cryptographic Overhead per Update Bundle")
    ax2.legend(loc="upper left")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, total.max() * 1.25)

    fig.suptitle(f"Security Level Comparison  (12 KB model delta, {d['iterations']} iterations)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig("fig2_security_levels.png")
    plt.close(fig)
    print("Saved fig2_security_levels.png")


# ── Figure 3: Byzantine Defense ────────────────────────────────────────────────

def fig3_byzantine(multiseed_path="results_multiseed.json"):
    # Pick the latest 20-round Byzantine results file
    candidates = sorted(glob.glob("results_byzantine_both_20r_*.json"))
    if not candidates:
        candidates = sorted(glob.glob("results_byzantine_both_*.json"))
    if not candidates:
        print("No Byzantine results found — skipping fig3")
        return
    byz_path = candidates[-1]
    print(f"  Using Byzantine file: {os.path.basename(byz_path)}")

    byz = json.load(open(byz_path))
    ms  = json.load(open(multiseed_path))

    n_rounds_byz = byz["rounds"]

    # Clean baseline from multiseed (use first n_rounds_byz rounds, mean only)
    by_round = ms["by_round"]
    clean_rounds = list(range(1, n_rounds_byz + 1))
    clean_auc = np.array([by_round[str(r)]["auc"]["mean"] for r in clean_rounds
                          if str(r) in by_round])
    clean_rounds = clean_rounds[:len(clean_auc)]

    # Part B: undefended vs defended
    byz_rounds    = [r["round"] for r in byz["part_b"]["without_detection"]["rounds"]]
    auc_undefended = np.array([r["auc"] for r in byz["part_b"]["without_detection"]["rounds"]])
    auc_defended   = np.array([r["auc"] for r in byz["part_b"]["with_detection"]["rounds"]])

    # Detection indicators for Part A (all True = 100%)
    part_a_detected = [r["detected"] for r in byz["part_a"]["rounds"]]
    detection_rate  = sum(part_a_detected) / len(part_a_detected)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    # ── Left: Part A — Signature Attack ─────────────────────────────
    part_a_auc = [r["auc"] for r in byz["part_a"]["rounds"]]
    ax1.plot(byz_rounds, part_a_auc, color=RED, lw=2, marker="o", ms=4,
             label="AUC (sig attack active)")
    ax1.plot(clean_rounds, clean_auc, color=BLUE, lw=2, ls="--",
             label="Clean baseline (no attack)")

    for rnd, detected in zip(byz_rounds, part_a_detected):
        ax1.axvline(rnd, color=GREEN, lw=0.5, alpha=0.4)

    ax1.set_xlabel("Communication Round")
    ax1.set_ylabel("Global AUC")
    ax1.set_title(f"Part A — Tampered Signature Attack\n"
                  f"ML-DSA detection rate: {detection_rate*100:.0f}%  (rogue=node {byz['rogue_node']})")
    ax1.set_ylim(0.2, 1.0)
    ax1.set_xlim(1, n_rounds_byz)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Green band annotation: "Rogue rejected every round"
    ax1.text(0.97, 0.06,
             f"Rogue node rejected\nin {int(detection_rate*100)}% of rounds",
             transform=ax1.transAxes, ha="right", va="bottom",
             fontsize=9, color=GREEN,
             bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=GREEN, alpha=0.8))

    # ── Right: Part B — Gradient Poisoning ──────────────────────────
    ax2.plot(clean_rounds, clean_auc, color=BLUE, lw=2, ls="--",
             label="Clean baseline (no attack)")
    ax2.plot(byz_rounds, auc_undefended, color=RED, lw=2, marker="s", ms=4,
             label=f"Under attack, no defense  (final={auc_undefended[-1]:.3f})")
    ax2.plot(byz_rounds, auc_defended, color=GREEN, lw=2, marker="^", ms=4,
             label=f"Under attack + norm filter (final={auc_defended[-1]:.3f})")

    # Annotate detection events for defended case
    for r in byz["part_b"]["with_detection"]["rounds"]:
        if r["detected"]:
            ax2.axvline(r["round"], color=GREEN, lw=0.5, alpha=0.3)

    b_det_rate = byz["part_b"]["with_detection"]["detection_rate"]
    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Global AUC")
    ax2.set_title(f"Part B — Gradient Poisoning Attack  (scale={byz['part_b']['poison_scale']}×)\n"
                  f"Norm-filter detection rate: {b_det_rate*100:.0f}%")
    ax2.set_ylim(0.2, 1.0)
    ax2.set_xlim(1, n_rounds_byz)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig("fig3_byzantine.png")
    plt.close(fig)
    print("Saved fig3_byzantine.png")


# ── Figure 4: PQC vs Classical Latency Breakdown ──────────────────────────────

def fig4_latency_breakdown():
    candidates = sorted(glob.glob("results_both_30r_*.json"))
    if not candidates:
        print("No PQC-vs-classical results found — skipping fig4")
        return
    path = candidates[-1]
    print(f"  Using comparison file: {os.path.basename(path)}")

    d = json.load(open(path))
    rd = d["rounds_detail"]
    pqc_rounds = rd["pqc"]
    cls_rounds = rd["classical"]

    rounds_x    = [r["round"] for r in pqc_rounds]
    pqc_enc     = np.array([r["avg_enc_ms"] for r in pqc_rounds])
    pqc_dec     = np.array([r["dec_ms"]     for r in pqc_rounds])
    pqc_total   = np.array([r["round_ms"]   for r in pqc_rounds])

    cls_enc     = np.array([r["avg_enc_ms"] for r in cls_rounds])
    cls_dec     = np.array([r["dec_ms"]     for r in cls_rounds])
    cls_total   = np.array([r["round_ms"]   for r in cls_rounds])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=False)

    def _band(ax, x, y, color, label):
        ax.plot(x, y, color=color, lw=2, label=label)
        ax.fill_between(x, y * 0.93, y * 1.07, color=color, alpha=0.15)

    # Panel 1: Encrypt time
    _band(axes[0], rounds_x, pqc_enc, BLUE,  "PQC  (ML-KEM-768 + ML-DSA-65)")
    _band(axes[0], rounds_x, cls_enc, RED,   "Classical  (RSA-4096 + HMAC)")
    axes[0].set_title("Encrypt / Sign (ms)\nper node per round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Time (ms)")

    # Panel 2: Decrypt time
    _band(axes[1], rounds_x, pqc_dec, BLUE, "PQC")
    _band(axes[1], rounds_x, cls_dec, RED,  "Classical")
    axes[1].set_title("Decrypt / Verify (ms)\nserver per round")
    axes[1].set_xlabel("Round")

    # Panel 3: Total round time
    _band(axes[2], rounds_x, pqc_total, BLUE, f"PQC  (mean {pqc_total.mean():.1f} ms)")
    _band(axes[2], rounds_x, cls_total, RED,  f"Classical  (mean {cls_total.mean():.1f} ms)")
    axes[2].set_title("Total Round Latency (ms)\n(training + crypto)")
    axes[2].set_xlabel("Round")

    ratio = cls_total.mean() / pqc_total.mean()
    axes[2].text(0.97, 0.06,
                 f"PQC is {ratio:.1f}× faster\nthan RSA per round",
                 transform=axes[2].transAxes, ha="right", va="bottom",
                 fontsize=9, color=BLUE,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=BLUE, alpha=0.8))

    for ax in axes:
        ax.legend(fontsize=8.5)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(1, max(rounds_x))

    fig.suptitle("PQC vs Classical Cryptography — Per-Round Latency  (5 nodes, 30 rounds)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig("fig4_latency_breakdown.png")
    plt.close(fig)
    print("Saved fig4_latency_breakdown.png")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    print("Generating figures...\n")
    fig1_convergence()
    fig2_security_levels()
    fig3_byzantine()
    fig4_latency_breakdown()
    print("\nDone.")
