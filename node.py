#!/usr/bin/env python3
"""EdgeNode: generates fake gradients, encrypts and signs them for the aggregation server."""

import os
import numpy as np
from dataclasses import dataclass

from pqc_layer import (
    KEMPublicKey,
    SigKeyPair,
    EncryptedPackage,
    TimingMs,
    encrypt_and_sign,
)

# Gradient shape: 3-layer MLP for NIDS (input→hidden→hidden→output weights flattened)
# 78 features (KDD99) → 64 → 32 → 5 classes  =  78*64 + 64*32 + 32*5 = 5152 + 2048 + 160 = 7360 params
GRADIENT_SHAPE = (7360,)
GRADIENT_DTYPE = np.float32
@dataclass
class NodeUpdate:
    package: EncryptedPackage
    timing: TimingMs
class EdgeNode:
    def __init__(self, node_id: int, server_kem_pub: KEMPublicKey, seed: int | None = None) -> None:
        self.node_id = node_id
        self._server_kem_pub = server_kem_pub
        self._sig_keypair = SigKeyPair()
        self._rng = np.random.default_rng(seed if seed is not None else node_id * 42 + 7)
        # Each node starts with a slightly different local model to simulate heterogeneous data
        self._local_model = self._rng.standard_normal(GRADIENT_SHAPE).astype(GRADIENT_DTYPE) * 0.1

    @property
    def sig_public_key(self):
        return self._sig_keypair.public_key

    def _compute_gradients(self) -> np.ndarray:
        """Simulate one round of local SGD: small noisy update around the local model."""
        noise = self._rng.standard_normal(GRADIENT_SHAPE).astype(GRADIENT_DTYPE) * 0.01
        gradient = -0.05 * self._local_model + noise
        self._local_model += gradient
        return gradient

    def prepare_update(self) -> NodeUpdate:
        """Compute gradients, serialize, encrypt and sign. Returns NodeUpdate with timings."""
        gradients = self._compute_gradients()
        payload = gradients.tobytes()
        package, timing = encrypt_and_sign(
            self.node_id, payload, self._server_kem_pub, self._sig_keypair
        )
        return NodeUpdate(package=package, timing=timing)
class RogueNode(EdgeNode):
    """Compromised node: sends poisoned gradients with a forged (invalid) signature."""

    def prepare_update(self) -> NodeUpdate:
        # Poisoned gradient: large constant to corrupt the global model
        poison = np.ones(GRADIENT_SHAPE, dtype=GRADIENT_DTYPE) * 100.0
        payload = poison.tobytes()

        # Encrypt legitimately so KEM decap + AES decrypt succeed on the server,
        # then swap in random bytes as the signature — ML-DSA-44 verify will fail.
        package, timing = encrypt_and_sign(
            self.node_id, payload, self._server_kem_pub, self._sig_keypair
        )
        forged_sig = os.urandom(len(package.signature))
        forged_package = EncryptedPackage(
            node_id=package.node_id,
            kem_ciphertext=package.kem_ciphertext,
            aes_ciphertext=package.aes_ciphertext,
            aes_nonce=package.aes_nonce,
            signature=forged_sig,
            payload_size=package.payload_size,
        )
        return NodeUpdate(package=forged_package, timing=timing)
