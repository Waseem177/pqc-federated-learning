#!/home/wasee/pqc-env/bin/python
"""AggServer: verifies, decrypts, and FedAvg-aggregates node updates."""

import sys

_VENV_SITE = "/home/wasee/pqc-env/lib/python3.12/site-packages"
if _VENV_SITE not in sys.path:
    sys.path.insert(0, _VENV_SITE)

import numpy as np
from dataclasses import dataclass, field

from pqc_layer import (
    KEMKeyPair,
    KEMPublicKey,
    SigPublicKey,
    TimingMs,
    decrypt_and_verify,
)
from node import NodeUpdate, GRADIENT_SHAPE, GRADIENT_DTYPE


@dataclass
class NodeRoundMetrics:
    node_id: int
    enc_timing: TimingMs          # from the node (passed through)
    dec_timing: TimingMs          # measured on the server
    payload_bytes: int
    overhead_bytes: int           # kem_ct + aes_ct + signature sizes


@dataclass
class RoundResult:
    round_num: int
    global_model: np.ndarray
    node_metrics: list[NodeRoundMetrics]

    @property
    def total_enc_ms(self) -> float:
        return sum(
            m.enc_timing.sign_ms + m.enc_timing.kem_encap_ms + m.enc_timing.aes_enc_ms
            for m in self.node_metrics
        )

    @property
    def total_dec_ms(self) -> float:
        return sum(
            m.dec_timing.kem_decap_ms + m.dec_timing.aes_dec_ms + m.dec_timing.sig_verify_ms
            for m in self.node_metrics
        )

    @property
    def avg_node_pqc_ms(self) -> float:
        """Average total PQC time per node this round (enc + dec)."""
        if not self.node_metrics:
            return 0.0
        return (self.total_enc_ms + self.total_dec_ms) / len(self.node_metrics)

    @property
    def total_overhead_bytes(self) -> int:
        return sum(m.overhead_bytes for m in self.node_metrics)


class AggServer:
    def __init__(self) -> None:
        self._kem_keypair = KEMKeyPair()
        self._node_sig_pubs: dict[int, SigPublicKey] = {}
        # Server maintains its own global model estimate
        self.global_model = np.zeros(GRADIENT_SHAPE, dtype=GRADIENT_DTYPE)

    @property
    def kem_public_key(self) -> KEMPublicKey:
        return self._kem_keypair.public_key

    def register_node(self, node_id: int, sig_pub: SigPublicKey) -> None:
        self._node_sig_pubs[node_id] = sig_pub

    def aggregate(self, updates: list[NodeUpdate], round_num: int) -> RoundResult:
        """Verify, decrypt, and FedAvg all node updates for one round."""
        gradients: list[np.ndarray] = []
        metrics: list[NodeRoundMetrics] = []
        rejected: list[int] = []

        for update in updates:
            pkg = update.package
            node_id = pkg.node_id

            if node_id not in self._node_sig_pubs:
                print(f"  [server] WARNING: unknown node {node_id}, skipping")
                rejected.append(node_id)
                continue

            try:
                plaintext, dec_timing = decrypt_and_verify(
                    pkg, self._kem_keypair, self._node_sig_pubs[node_id]
                )
            except ValueError as exc:
                print(f"  [server] REJECTED node {node_id}: {exc}")
                rejected.append(node_id)
                continue

            gradient = np.frombuffer(plaintext, dtype=GRADIENT_DTYPE).reshape(GRADIENT_SHAPE)
            gradients.append(gradient)

            overhead = len(pkg.kem_ciphertext) + len(pkg.aes_ciphertext) + len(pkg.signature)
            metrics.append(NodeRoundMetrics(
                node_id=node_id,
                enc_timing=update.timing,
                dec_timing=dec_timing,
                payload_bytes=pkg.payload_size,
                overhead_bytes=overhead,
            ))

        if not gradients:
            raise RuntimeError(f"Round {round_num}: all {len(updates)} updates rejected")

        # FedAvg: equal weights (uniform data distribution assumption)
        aggregated = np.mean(np.stack(gradients, axis=0), axis=0)
        self.global_model += aggregated

        return RoundResult(
            round_num=round_num,
            global_model=self.global_model.copy(),
            node_metrics=metrics,
        )
