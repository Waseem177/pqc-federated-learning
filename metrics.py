#!/home/wasee/pqc-env/bin/python
"""Metrics collection, aggregation, and CSV export for the FL simulation."""

import csv
import sys
from dataclasses import dataclass, field, asdict

_VENV_SITE = "/home/wasee/pqc-env/lib/python3.12/site-packages"
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

from server import RoundResult

RSA_BASELINE_MS = 71.0   # measured avg round latency with RSA (from prior benchmark)
PQC_LABEL = "Kyber512 + ML-DSA-44"


@dataclass
class RoundSummary:
    round_num: int
    num_nodes: int
    avg_pqc_ms: float        # avg total PQC time per node this round
    total_enc_ms: float      # sum of all node encryption times
    total_dec_ms: float      # sum of all server decryption/verify times
    total_overhead_bytes: int


class MetricsCollector:
    def __init__(self) -> None:
        self._rounds: list[RoundSummary] = []

    def record(self, result: RoundResult) -> RoundSummary:
        summary = RoundSummary(
            round_num=result.round_num,
            num_nodes=len(result.node_metrics),
            avg_pqc_ms=result.avg_node_pqc_ms,
            total_enc_ms=result.total_enc_ms,
            total_dec_ms=result.total_dec_ms,
            total_overhead_bytes=result.total_overhead_bytes,
        )
        self._rounds.append(summary)
        return summary

    def __len__(self) -> int:
        return len(self._rounds)

    def avg_pqc_ms(self) -> float:
        if not self._rounds:
            return 0.0
        return sum(r.avg_pqc_ms for r in self._rounds) / len(self._rounds)

    def total_overhead_bytes(self) -> int:
        return sum(r.total_overhead_bytes for r in self._rounds)

    def speedup_vs_rsa(self) -> float:
        avg = self.avg_pqc_ms()
        return RSA_BASELINE_MS / avg if avg > 0 else float("inf")

    def per_node_overhead_kb(self) -> float:
        """Average per-node overhead across all rounds in KB."""
        if not self._rounds:
            return 0.0
        total_nodes = sum(r.num_nodes for r in self._rounds)
        total_bytes = sum(r.total_overhead_bytes for r in self._rounds)
        return (total_bytes / total_nodes) / 1024 if total_nodes else 0.0

    def export_csv(self, path: str) -> None:
        if not self._rounds:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(self._rounds[0]).keys()))
            writer.writeheader()
            writer.writerows(asdict(r) for r in self._rounds)
