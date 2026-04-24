# PQC-Secured Federated Learning NIDS

Federated learning simulation where five edge nodes train locally on network traffic data and send encrypted, authenticated model updates to a central server — secured with post-quantum cryptography instead of classical RSA/ECDH.

Built as a proof-of-concept for hardening FL pipelines against quantum adversaries before the threat becomes real.


## The Problem

RSA and ECDH break under Shor's algorithm on a sufficiently powerful quantum computer. For federated learning deployments, that means two concrete risks:

- **Harvest-now, decrypt-later** — adversaries archive encrypted model updates today and decrypt them once quantum hardware is available
- **Model poisoning via forged updates** — if you can break the signature scheme, you can inject malicious gradients

This project swaps out classical crypto for the NIST-standardized PQC algorithms (ML-KEM, ML-DSA) that are designed to hold up against both. The overhead is negligible — around 100× faster than RSA in practice.


## How It Works

Each node runs local SGD on a small MLP (78→64→32→5, ~7360 params, KDD99 feature space), then securely submits its update each round:

1. The gradient update gets **signed** with the node's ML-DSA-44 key
2. An AES-256-GCM session key is derived from a fresh Kyber512 KEM encapsulation against the server's public key
3. The signed update is **encrypted** and sent to the server
4. The server **decapsulates** the KEM ciphertext, **decrypts** the payload, and **verifies** the signature — only then does it accept the update
5. All verified updates are aggregated with FedAvg (uniform weights)


## PQC Stack

| Primitive | Algorithm | Standard | Purpose |
|-----------|-----------|----------|---------|
| KEM | ML-KEM / Kyber512 | NIST FIPS 203 | Per-round session key |
| Signatures | ML-DSA-44 | NIST FIPS 204 | Node identity + update integrity |
| Symmetric encryption | AES-256-GCM | NIST FIPS 197 | Payload encryption with KEM-derived key |


## Results

| Metric | PQC (this work) | RSA-2048 | Speedup |
|--------|----------------|----------|---------|
| Avg overhead / node / round | **0.71 ms** | 71 ms | ~100× |
| Total crypto overhead / round (5 nodes) | ~3.5 ms | ~355 ms | ~100× |
| Per-node payload overhead | 32.6 KB | ~0.5 KB | — |
| Signature failures across 10 rounds | 0 / 50 | — | — |

Benchmarked on Python 3.12, Ubuntu 22.04 WSL2, liboqs 0.10.x. Round 0 includes key-gen warm-up; steady-state (rounds 1–9) averages ~0.57 ms/node.


## Setup

**Prerequisites:** Python 3.12, liboqs (provides `oqs` bindings), `cryptography`

```bash
# Virtual env
python3.12 -m venv ~/pqc-env
source ~/pqc-env/bin/activate

# Dependencies
pip install liboqs-python cryptography numpy

# Clone
git clone https://github.com/Waseem177/pqc-federated-learning.git
cd pqc-federated-learning
```

If `liboqs-python` doesn't have a wheel for your platform, build from source: https://github.com/open-quantum-safe/liboqs-python


## Usage

```bash
# Run the simulation (5 nodes, 10 rounds by default)
~/pqc-env/bin/python simulation.py

# Custom parameters
~/pqc-env/bin/python simulation.py --rounds 20 --nodes 5 --csv my_results.csv
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds` | 10 | FL rounds |
| `--nodes` | 5 | Edge nodes |
| `--csv` | `results.csv` | Output file |

```bash
# Test the PQC layer standalone
~/pqc-env/bin/python pqc_layer.py
```

The simulation prints a per-round table to the terminal and exports a CSV with columns: `round_num, num_nodes, avg_pqc_ms, total_enc_ms, total_dec_ms, total_overhead_bytes`


## Project Structure

```
├── simulation.py   # Entry point — orchestrates nodes, server, and metrics
├── node.py         # EdgeNode: local training + PQC encrypt/sign
├── server.py       # AggServer: PQC decrypt/verify + FedAvg
├── pqc_layer.py    # Kyber512 KEM, ML-DSA-44 signatures, AES-GCM
├── metrics.py      # Per-round and aggregate metrics
├── dashboard.py    # Terminal output formatting
└── results.csv     # Sample benchmark output (10 rounds, 5 nodes)
```


## What's Next

- [x] Kyber512 KEM + ML-DSA-44 sign/verify
- [x] AES-256-GCM payload encryption
- [x] 5-node FL simulation with FedAvg
- [x] Per-round latency metrics + CSV export
- [ ] Real dataset integration (KDD99 / CICIDS2017)
- [ ] Differential privacy noise injection
- [ ] Docker containerization for reproducible multi-node deployment


## References

- [NIST FIPS 203 — ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 — ML-DSA](https://csrc.nist.gov/pubs/fips/204/final)
- [Open Quantum Safe / liboqs](https://openquantumsafe.org/)
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg, 2017)


MIT License
