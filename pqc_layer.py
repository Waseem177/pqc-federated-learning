#!/usr/bin/env python3
"""PQC primitives: Kyber512 KEM + ML-DSA-44 signatures + AES-GCM payload encryption."""

import os
import time
from dataclasses import dataclass
import oqs
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEM_ALG = "Kyber512"
SIG_ALG = "ML-DSA-44"
AES_NONCE_BYTES = 12
@dataclass
class KEMPublicKey:
    raw: bytes
@dataclass
class SigPublicKey:
    raw: bytes
@dataclass
class EncryptedPackage:
    """Everything the server needs to decrypt and verify one node update."""
    node_id: int
    kem_ciphertext: bytes   # Kyber512 encapsulation ciphertext
    aes_ciphertext: bytes   # AES-GCM ciphertext+tag
    aes_nonce: bytes
    signature: bytes        # ML-DSA-44 signature over aes_ciphertext
    payload_size: int       # original plaintext bytes, for metrics
@dataclass
class TimingMs:
    sign_ms: float = 0.0
    kem_encap_ms: float = 0.0
    aes_enc_ms: float = 0.0
    kem_decap_ms: float = 0.0
    aes_dec_ms: float = 0.0
    sig_verify_ms: float = 0.0
def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000
class KEMKeyPair:
    """Server-side Kyber512 KEM keypair. Generates once, used across all rounds."""

    def __init__(self) -> None:
        with oqs.KeyEncapsulation(KEM_ALG) as kem:
            self._pub = kem.generate_keypair()
            self._sec = kem.export_secret_key()

    @property
    def public_key(self) -> KEMPublicKey:
        return KEMPublicKey(self._pub)

    def decapsulate(self, ciphertext: bytes) -> tuple[bytes, float]:
        """Returns (shared_secret, elapsed_ms)."""
        t = time.perf_counter()
        with oqs.KeyEncapsulation(KEM_ALG, secret_key=self._sec) as kem:
            secret = kem.decap_secret(ciphertext)
        return secret, _ms(t)
class SigKeyPair:
    """Per-node ML-DSA-44 signing keypair."""

    def __init__(self) -> None:
        with oqs.Signature(SIG_ALG) as sig:
            self._pub = sig.generate_keypair()
            self._sec = sig.export_secret_key()

    @property
    def public_key(self) -> SigPublicKey:
        return SigPublicKey(self._pub)

    def sign(self, message: bytes) -> tuple[bytes, float]:
        """Returns (signature, elapsed_ms)."""
        t = time.perf_counter()
        with oqs.Signature(SIG_ALG, secret_key=self._sec) as sig:
            signature = sig.sign(message)
        return signature, _ms(t)
def kem_encapsulate(server_pub: KEMPublicKey) -> tuple[bytes, bytes, float]:
    """Returns (kem_ciphertext, shared_secret, elapsed_ms)."""
    t = time.perf_counter()
    with oqs.KeyEncapsulation(KEM_ALG) as kem:
        ciphertext, shared_secret = kem.encap_secret(server_pub.raw)
    return ciphertext, shared_secret, _ms(t)
def sig_verify(message: bytes, signature: bytes, pub: SigPublicKey) -> tuple[bool, float]:
    """Returns (valid, elapsed_ms)."""
    t = time.perf_counter()
    with oqs.Signature(SIG_ALG) as sig:
        valid = sig.verify(message, signature, pub.raw)
    return valid, _ms(t)
def aes_encrypt(plaintext: bytes, shared_secret: bytes) -> tuple[bytes, bytes, float]:
    """AES-256-GCM encrypt. Returns (ciphertext_with_tag, nonce, elapsed_ms)."""
    key = shared_secret[:32]
    nonce = os.urandom(AES_NONCE_BYTES)
    t = time.perf_counter()
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return ciphertext, nonce, _ms(t)
def aes_decrypt(ciphertext: bytes, shared_secret: bytes, nonce: bytes) -> tuple[bytes, float]:
    """AES-256-GCM decrypt. Returns (plaintext, elapsed_ms). Raises on auth failure."""
    key = shared_secret[:32]
    t = time.perf_counter()
    plaintext = AESGCM(key).decrypt(nonce, ciphertext, None)
    return plaintext, _ms(t)
def encrypt_and_sign(
    node_id: int,
    payload: bytes,
    server_kem_pub: KEMPublicKey,
    node_sig_keypair: SigKeyPair,
) -> tuple[EncryptedPackage, TimingMs]:
    """Full node-side PQC pipeline: sign → KEM encap → AES-GCM encrypt."""
    timing = TimingMs()

    signature, timing.sign_ms = node_sig_keypair.sign(payload)
    kem_ct, shared_secret, timing.kem_encap_ms = kem_encapsulate(server_kem_pub)
    aes_ct, nonce, timing.aes_enc_ms = aes_encrypt(payload, shared_secret)

    pkg = EncryptedPackage(
        node_id=node_id,
        kem_ciphertext=kem_ct,
        aes_ciphertext=aes_ct,
        aes_nonce=nonce,
        signature=signature,
        payload_size=len(payload),
    )
    return pkg, timing
def decrypt_and_verify(
    pkg: EncryptedPackage,
    server_kem_keypair: KEMKeyPair,
    node_sig_pub: SigPublicKey,
) -> tuple[bytes, TimingMs]:
    """Full server-side PQC pipeline: KEM decap → AES-GCM decrypt → verify.

    Raises ValueError if signature check fails.
    """
    timing = TimingMs()

    shared_secret, timing.kem_decap_ms = server_kem_keypair.decapsulate(pkg.kem_ciphertext)
    plaintext, timing.aes_dec_ms = aes_decrypt(pkg.aes_ciphertext, shared_secret, pkg.aes_nonce)
    valid, timing.sig_verify_ms = sig_verify(plaintext, pkg.signature, node_sig_pub)

    if not valid:
        raise ValueError(f"Signature verification failed for node {pkg.node_id}")

    return plaintext, timing
if __name__ == "__main__":
    print("=== pqc_layer self-test ===")

    server_kem = KEMKeyPair()
    node_sig = SigKeyPair()
    payload = b"gradient-data-test-" * 50

    pkg, enc_timing = encrypt_and_sign(0, payload, server_kem.public_key, node_sig)
    recovered, dec_timing = decrypt_and_verify(pkg, server_kem, node_sig.public_key)

    assert recovered == payload, "Payload mismatch!"
    print(f"  sign:       {enc_timing.sign_ms:.3f} ms")
    print(f"  kem_encap:  {enc_timing.kem_encap_ms:.3f} ms")
    print(f"  aes_enc:    {enc_timing.aes_enc_ms:.3f} ms")
    print(f"  kem_decap:  {dec_timing.kem_decap_ms:.3f} ms")
    print(f"  aes_dec:    {dec_timing.aes_dec_ms:.3f} ms")
    print(f"  sig_verify: {dec_timing.sig_verify_ms:.3f} ms")
    total = sum([
        enc_timing.sign_ms, enc_timing.kem_encap_ms, enc_timing.aes_enc_ms,
        dec_timing.kem_decap_ms, dec_timing.aes_dec_ms, dec_timing.sig_verify_ms,
    ])
    print(f"  total:      {total:.3f} ms")
    overhead = len(pkg.kem_ciphertext) + len(pkg.aes_ciphertext) + len(pkg.signature)
    print(f"  overhead:   {overhead} bytes  (payload: {len(payload)} bytes)")
    print("PASS")
