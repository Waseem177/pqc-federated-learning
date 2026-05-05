import os, time, hmac, hashlib
from dataclasses import dataclass
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

AES_NONCE_LEN = 12

def _ms(t0): return (time.perf_counter() - t0) * 1000.0

@dataclass
class ClassicalPackage:
    node_id: int
    rsa_ciphertext: bytes
    aes_ciphertext: bytes
    aes_nonce: bytes
    hmac_tag: bytes
    payload_size: int

    def total_bytes(self):
        return len(self.rsa_ciphertext)+len(self.aes_ciphertext)+len(self.aes_nonce)+len(self.hmac_tag)

    def overhead_bytes(self):
        return self.total_bytes() - self.payload_size

@dataclass
class ClassicalTimingMs:
    rsa_enc_ms: float = 0.0
    aes_enc_ms: float = 0.0
    hmac_ms: float = 0.0
    rsa_dec_ms: float = 0.0
    aes_dec_ms: float = 0.0
    hmac_verify_ms: float = 0.0

    def total_enc_ms(self): return self.rsa_enc_ms + self.aes_enc_ms + self.hmac_ms
    def total_dec_ms(self): return self.rsa_dec_ms + self.aes_dec_ms + self.hmac_verify_ms

class RSAKeyPair:
    def __init__(self):
        print("  Generating RSA-4096 keypair (slow, once only)...")
        self._priv = rsa.generate_private_key(public_exponent=65537, key_size=4096)
        self._pub = self._priv.public_key()

    @property
    def public_key(self): return self._pub

    def decrypt(self, ct):
        t = time.perf_counter()
        pt = self._priv.decrypt(ct, padding.OAEP(
            mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
        return pt, _ms(t)

def classical_encrypt(node_id, payload, server_pub, hmac_key):
    t = ClassicalTimingMs()
    aes_key = os.urandom(32)
    t0 = time.perf_counter()
    rsa_ct = server_pub.encrypt(aes_key, padding.OAEP(
        mgf=padding.MGF1(hashes.SHA256()), algorithm=hashes.SHA256(), label=None))
    t.rsa_enc_ms = _ms(t0)
    nonce = os.urandom(AES_NONCE_LEN)
    t0 = time.perf_counter()
    aes_ct = AESGCM(aes_key).encrypt(nonce, payload, None)
    t.aes_enc_ms = _ms(t0)
    t0 = time.perf_counter()
    tag = hmac.new(hmac_key, rsa_ct + aes_ct, hashlib.sha256).digest()
    t.hmac_ms = _ms(t0)
    return ClassicalPackage(node_id, rsa_ct, aes_ct, nonce, tag, len(payload)), t

def classical_decrypt(pkg, server_kp, hmac_key):
    t = ClassicalTimingMs()
    t0 = time.perf_counter()
    expected = hmac.new(hmac_key, pkg.rsa_ciphertext + pkg.aes_ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(expected, pkg.hmac_tag):
        raise ValueError(f"HMAC FAILED for node {pkg.node_id}")
    t.hmac_verify_ms = _ms(t0)
    aes_key, t.rsa_dec_ms = server_kp.decrypt(pkg.rsa_ciphertext)
    t0 = time.perf_counter()
    pt = AESGCM(aes_key).decrypt(pkg.aes_nonce, pkg.aes_ciphertext, None)
    t.aes_dec_ms = _ms(t0)
    return pt, t

if __name__ == "__main__":
    print("=== Classical Baseline (RSA-4096 + AES-256-GCM + HMAC-SHA256) ===")
    server = RSAKeyPair()
    hmac_key = os.urandom(32)
    payload = os.urandom(4096)
    pkg, enc_t = classical_encrypt(0, payload, server.public_key, hmac_key)
    recovered, dec_t = classical_decrypt(pkg, server, hmac_key)
    assert recovered == payload
    print(f"  RSA-4096 enc  : {enc_t.rsa_enc_ms:.3f} ms")
    print(f"  AES enc       : {enc_t.aes_enc_ms:.3f} ms")
    print(f"  HMAC          : {enc_t.hmac_ms:.3f} ms")
    print(f"  RSA-4096 dec  : {dec_t.rsa_dec_ms:.3f} ms")
    print(f"  AES dec       : {dec_t.aes_dec_ms:.3f} ms")
    print(f"  HMAC verify   : {dec_t.hmac_verify_ms:.3f} ms")
    print(f"  Bundle size   : {pkg.total_bytes()/1024:.2f} KB  (overhead: {pkg.overhead_bytes()} B)")
    print("PASS")
