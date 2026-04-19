# PQC-Secured Federated Learning NIDS

A simulation of **post-quantum cryptography (PQC) secured federated learning** for network intrusion detection. Five edge nodes train locally on network traffic data and submit encrypted, authenticated model updates to a central aggregation server — hardened against both classical and quantum adversaries.

---

## Why PQC Matters

Classical public-key cryptography (RSA, ECDH) is broken by Shor's algorithm running on a sufficiently powerful quantum computer. As quantum hardware matures, any RSA-protected federated learning deployment becomes vulnerable to:

- **Harvest-now, decrypt-later** attacks — adversaries archive today's encrypted model updates to decrypt once a quantum computer is available.
- **Model poisoning** — forged updates injected by an adversary who can break classical signature schemes.

This project replaces classical crypto with **NIST-standardized post-quantum algorithms** that are believed to be secure against both classical and quantum attacks, with negligible performance overhead compared to RSA.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Aggregation Server                  │
│  Kyber512 KEM keypair  ·  FedAvg  ·  Sig registry   │
└────────────────┬──────────────────────────────────── ┘
                 │  EncryptedPackage (per node, per round)
   ┌─────────────┼─────────────┐
   │             │             │  ...  (5 nodes total)
   ▼             ▼             ▼
Node 0         Node 1        Node 4
ML-DSA-44     ML-DSA-44    ML-DSA-44
sign           sign          sign
   │             │             │
Kyber512      Kyber512     Kyber512
KEM encap     KEM encap    KEM encap
   │             │             │
AES-256-GCM   AES-256-GCM  AES-256-GCM
encrypt        encrypt       encrypt
```

**Per-round flow:**
1. Each edge node simulates local SGD on a 7360-parameter MLP (KDD99 feature space: 78→64→32→5).
2. The gradient update is **signed** with the node's ML-DSA-44 key, then **encrypted** with an AES-256-GCM key derived from a fresh Kyber512 KEM encapsulation.
3. The server **decapsulates** the KEM ciphertext, **decrypts** the payload, and **verifies** the signature before accepting the update.
4. Verified updates are aggregated via **FedAvg** (uniform weights).

---

## PQC Stack

| Primitive | Algorithm | Standard | Purpose |
|-----------|-----------|----------|---------|
| Key Encapsulation (KEM) | ML-KEM / Kyber512 | NIST FIPS 203 | Confidentiality — per-round session key |
| Digital Signature | ML-DSA-44 | NIST FIPS 204 | Authentication — node identity + update integrity |
| Symmetric Encryption | AES-256-GCM | NIST FIPS 197 | Payload encryption using KEM-derived key |

---

## Key Results

| Metric | PQC (this work) | RSA-2048 baseline | Speedup |
|--------|----------------|-------------------|---------|
| Avg PQC overhead / node / round | **0.71 ms** | 71 ms | **~100×** |
| Crypto overhead / round (5 nodes) | ~3.5 ms total | ~355 ms total | ~100× |
| Per-node payload overhead | 32.6 KB | ~0.5 KB | — |
| Signature failures (10 rounds) | 0 / 50 | — | — |

> Benchmarked on Python 3.12, Ubuntu 22.04 WSL2, liboqs 0.10.x.  
> Round 0 includes key-generation warm-up; steady-state (rounds 1–9) averages ~0.57 ms/node.

---

## Installation

### Prerequisites

- Python 3.12
- liboqs built and installed (provides `oqs` Python bindings)
- `cryptography` library

### 1. Set up the virtual environment

```bash
python3.12 -m venv ~/pqc-env
source ~/pqc-env/bin/activate
```

### 2. Install liboqs-python

```bash
pip install liboqs-python
# or build from source if the wheel is unavailable for your platform:
# https://github.com/open-quantum-safe/liboqs-python
```

### 3. Install remaining dependencies

```bash
pip install cryptography numpy
```

### 4. Clone this repository

```bash
git clone https://github.com/Waseem177/pqc-fl-hackathon.git
cd pqc-fl-hackathon
```

---

## Usage

### Run the full simulation (default: 5 nodes, 10 rounds)

```bash
~/pqc-env/bin/python simulation.py
```

### Custom parameters

```bash
~/pqc-env/bin/python simulation.py --rounds 20 --nodes 5 --csv my_results.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds` | 10 | Number of federated learning rounds |
| `--nodes` | 5 | Number of simulated edge nodes |
| `--csv` | `results.csv` | Output path for per-round metrics |

### Verify the PQC layer standalone

```bash
~/pqc-env/bin/python pqc_layer.py
```

---

## Output

The dashboard prints a per-round table and final summary to the terminal, and exports a CSV with columns:

```
round_num, num_nodes, avg_pqc_ms, total_enc_ms, total_dec_ms, total_overhead_bytes
```

---

## Project Structure

```
pqc-fl-hackathon/
├── simulation.py   # Entry point — orchestrates nodes, server, and metrics
├── node.py         # EdgeNode: local gradient simulation + PQC encrypt/sign
├── server.py       # AggServer: PQC decrypt/verify + FedAvg aggregation
├── pqc_layer.py    # PQC primitives: Kyber512 KEM, ML-DSA-44 signatures, AES-GCM
├── metrics.py      # Per-round and aggregate metric collection
├── dashboard.py    # Terminal output formatting
└── results.csv     # Sample benchmark results (10 rounds, 5 nodes)
```

---

## Roadmap

- [x] Kyber512 KEM key exchange
- [x] ML-DSA-44 signing and verification
- [x] AES-256-GCM payload encryption
- [x] 5-node FL simulation with FedAvg aggregation
- [x] Per-round latency and overhead metrics
- [x] CSV export and terminal dashboard
- [ ] Real network intrusion dataset (KDD99 / CICIDS2017) integration
- [ ] Differential privacy noise injection
- [ ] **Docker containerization** — planned for the full research paper implementation to enable reproducible multi-node deployment across isolated containers

---

## References

- [NIST FIPS 203 — ML-KEM (Kyber)](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 — ML-DSA (Dilithium)](https://csrc.nist.gov/pubs/fips/204/final)
- [Open Quantum Safe / liboqs](https://openquantumsafe.org/)
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg, 2017)

---

## License

MIT
