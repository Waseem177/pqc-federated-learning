import os, sys, time
from dataclasses import dataclass
import oqs
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEM_ALG = "ML-KEM-768"
SIG_ALG = "ML-DSA-65"
AES_NONCE_LEN = 12
AES_KEY_LEN = 32
HKDF_INFO = b"pq-fedhealth-v1"

def _ms(t0): return (time.perf_counter() - t0) * 1000.0

def _hkdf(ss):
    return HKDF(algorithm=hashes.SHA256(), length=AES_KEY_LEN, salt=None, info=HKDF_INFO).derive(ss)

@dataclass
class KEMPublicKey:
    raw: bytes

@dataclass
class EncryptedPackage:
    node_id: int
    kem_ciphertext: bytes
    aes_ciphertext: bytes
    aes_nonce: bytes
    signature: bytes
    sig_public_key: bytes
    payload_size: int

    def total_bytes(self):
        return len(self.kem_ciphertext)+len(self.aes_ciphertext)+len(self.aes_nonce)+len(self.signature)+len(self.sig_public_key)

    def overhead_bytes(self):
        return self.total_bytes() - self.payload_size

@dataclass
class TimingMs:
    sign_ms: float = 0.0
    kem_encap_ms: float = 0.0
    aes_enc_ms: float = 0.0
    kem_decap_ms: float = 0.0
    aes_dec_ms: float = 0.0
    sig_verify_ms: float = 0.0

    def total_enc_ms(self): return self.sign_ms + self.kem_encap_ms + self.aes_enc_ms
    def total_dec_ms(self): return self.kem_decap_ms + self.aes_dec_ms + self.sig_verify_ms

class KEMKeyPair:
    def __init__(self):
        with oqs.KeyEncapsulation(KEM_ALG) as kem:
            self._pub = kem.generate_keypair()
            self._sec = kem.export_secret_key()

    @property
    def public_key(self): return KEMPublicKey(self._pub)

    def decapsulate(self, ct):
        t = time.perf_counter()
        with oqs.KeyEncapsulation(KEM_ALG, secret_key=self._sec) as kem:
            ss = kem.decap_secret(ct)
        return ss, _ms(t)

class SigKeyPair:
    def __init__(self):
        with oqs.Signature(SIG_ALG) as sig:
            self._pub = sig.generate_keypair()
            self._sec = sig.export_secret_key()

    @property
    def public_key_bytes(self): return self._pub

    def sign(self, msg):
        t = time.perf_counter()
        with oqs.Signature(SIG_ALG, secret_key=self._sec) as sig:
            s = sig.sign(msg)
        return s, _ms(t)

def kem_encapsulate(server_pub):
    t = time.perf_counter()
    with oqs.KeyEncapsulation(KEM_ALG) as kem:
        ct, ss = kem.encap_secret(server_pub.raw)
    return ct, ss, _ms(t)

def sig_verify(msg, signature, pub):
    t = time.perf_counter()
    with oqs.Signature(SIG_ALG) as sig:
        valid = sig.verify(msg, signature, pub)
    return valid, _ms(t)

def aes_encrypt(pt, ss):
    key = _hkdf(ss)
    nonce = os.urandom(AES_NONCE_LEN)
    t = time.perf_counter()
    ct = AESGCM(key).encrypt(nonce, pt, None)
    return ct, nonce, _ms(t)

def aes_decrypt(ct, ss, nonce):
    key = _hkdf(ss)
    t = time.perf_counter()
    pt = AESGCM(key).decrypt(nonce, ct, None)
    return pt, _ms(t)

def encrypt_and_sign(node_id, payload, server_kem_pub, node_sig_kp):
    t = TimingMs()
    kem_ct, ss, t.kem_encap_ms = kem_encapsulate(server_kem_pub)
    aes_ct, nonce, t.aes_enc_ms = aes_encrypt(payload, ss)
    sig, t.sign_ms = node_sig_kp.sign(kem_ct + aes_ct)
    pkg = EncryptedPackage(node_id, kem_ct, aes_ct, nonce, sig, node_sig_kp.public_key_bytes, len(payload))
    return pkg, t

def decrypt_and_verify(pkg, server_kem_kp):
    t = TimingMs()
    ss, t.kem_decap_ms = server_kem_kp.decapsulate(pkg.kem_ciphertext)
    pt, t.aes_dec_ms = aes_decrypt(pkg.aes_ciphertext, ss, pkg.aes_nonce)
    valid, t.sig_verify_ms = sig_verify(pkg.kem_ciphertext + pkg.aes_ciphertext, pkg.signature, pkg.sig_public_key)
    if not valid:
        raise ValueError(f"Signature FAILED for node {pkg.node_id}")
    return pt, t

if __name__ == "__main__":
    print("=== PQ-FedHealth PQC Layer ===")
    print(f"KEM: {KEM_ALG}  SIG: {SIG_ALG}")
    server_kem = KEMKeyPair()
    node_sig = SigKeyPair()
    payload = os.urandom(4096)
    pkg, enc_t = encrypt_and_sign(0, payload, server_kem.public_key, node_sig)
    recovered, dec_t = decrypt_and_verify(pkg, server_kem)
    assert recovered == payload
    print(f"  KEM encap   : {enc_t.kem_encap_ms:.3f} ms")
    print(f"  AES enc     : {enc_t.aes_enc_ms:.3f} ms")
    print(f"  Sign        : {enc_t.sign_ms:.3f} ms")
    print(f"  KEM decap   : {dec_t.kem_decap_ms:.3f} ms")
    print(f"  AES dec     : {dec_t.aes_dec_ms:.3f} ms")
    print(f"  Verify      : {dec_t.sig_verify_ms:.3f} ms")
    print(f"  Bundle size : {pkg.total_bytes()/1024:.2f} KB  (overhead: {pkg.overhead_bytes()} B)")
    print("PASS")
