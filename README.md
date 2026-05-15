# PQC-Secured Federated Learning for Healthcare

A working prototype for securing federated learning model updates with post-quantum cryptography across a simulated multi-hospital network. Five hospital nodes train locally on real medical datasets — diabetes, heart disease, and breast cancer — then submit encrypted and signed weight updates to a central aggregation server.

Rather than claiming a full production system, this repository focuses on the security layer: how model updates can be confidentially transmitted, authenticated, and verified before aggregation in a quantum-resilient setting.


## The Problem

RSA and ECDH break under Shor's algorithm on a sufficiently powerful quantum computer. For federated learning in healthcare, that means two concrete risks:

- **Harvest-now, decrypt-later** — adversaries archive encrypted model updates today and decrypt them once quantum hardware is available
- **Model poisoning via forged updates** — if you can break the signature scheme, you can inject malicious gradients into the aggregated model

This project swaps out classical crypto for the NIST-standardized PQC algorithms (ML-KEM, ML-DSA) that are designed to hold up against both.


## How It Works

Each of the 5 hospital nodes trains a local feedforward neural network on its own dataset, then securely submits its gradient update each round:

1. The gradient update gets **signed** with the node's ML-DSA-65 key
2. An AES-256-GCM session key is derived from a fresh ML-KEM-768 encapsulation against the server's public key
3. The signed update is **encrypted** and sent to the server
4. The server **decapsulates** the KEM ciphertext, **decrypts** the payload, and **verifies** the signature — only then does it accept the update
5. All verified updates are aggregated with FedAvg (weighted by local training set size)


## Dataset Setup (Non-IID Multi-Hospital)

Five hospital nodes — each with a different dataset and class distribution — simulate a realistic federated healthcare scenario:

| Node | Dataset | Task | Train | Test |
|------|---------|------|-------|------|
| 0 | PIMA Diabetes (shard A) | Diabetes prediction | 307 | 153 |
| 1 | PIMA Diabetes (shard B) | Diabetes prediction | 308 | 153 |
| 2 | Cleveland Heart Disease (UCI) | Heart disease detection | 238 | 59 |
| 3 | WDBC Breast Cancer (UCI) | Malignancy classification | 456 | 113 |
| 4 | PIMA + Gaussian noise | Diabetes (noisy clinic) | 615 | 153 |

All features are normalized per-node (simulating local calibration). PIMA (8 features) and Cleveland (13 features) are zero-padded to WDBC dimensionality (30 features). Global test set: union of all hospital test sets (631 samples).

Model: 3-layer feedforward NN (30 → 64 → 32 → 1, ReLU activations, sigmoid output, binary cross-entropy loss).


## PQC Stack

| Primitive | Algorithm | Standard | Purpose |
|-----------|-----------|----------|---------|
| KEM | ML-KEM-768 | NIST FIPS 203 | Per-round session key |
| Signatures | ML-DSA-65 | NIST FIPS 204 | Node identity + update integrity |
| Symmetric encryption | AES-256-GCM | NIST FIPS 197 | Payload encryption with KEM-derived key |

Classical baseline for comparison: RSA-4096 (key encapsulation) + HMAC-SHA256 + AES-256-GCM.


## Results

30-round benchmark, 5 nodes, lr=0.01. Benchmarked on Python 3.12, Ubuntu WSL2 (Linux 6.6 kernel), no GPU, liboqs 0.15.0.

**Model quality — identical across both security schemes:**

| Metric | PQC | Classical |
|--------|-----|-----------|
| AUC | **0.837** | 0.837 |
| Sensitivity (TPR) | 0.531 | 0.526 |
| Specificity (TNR) | 0.901 | 0.913 |
| F1 Score | 0.622 | 0.627 |
| Accuracy | 76.7% | 77.3% |

**Security overhead:**

| Metric | PQC | Classical |
|--------|-----|-----------|
| Avg enc per node | 0.57 ms | 0.29 ms |
| Avg dec per round | **1.41 ms** | 33.44 ms |
| Avg round time | **64 ms** | 103 ms |
| Crypto overhead per round | 31.9 KB | 2.9 KB |
| Total network (30 rounds) | 3381.2 KB | 2530.8 KB |
| Signature verification failures | 0 / 150 | — |

Key takeaway: PQC decryption is **24× faster** than RSA-4096, and rounds are **1.6× faster** end-to-end. The wire overhead is larger (KEM ciphertext + DSA signatures), but negligible on any modern network. Model quality is identical — the PQC layer adds no utility cost.


## Setup

**Prerequisites:** Python 3.12, liboqs, `cryptography`, `numpy`

```bash
python3.12 -m venv ~/pqc-env
source ~/pqc-env/bin/activate
pip install liboqs-python cryptography numpy

git clone https://github.com/Waseem177/pqc-federated-learning.git
cd pqc-federated-learning
```

If `liboqs-python` doesn't have a wheel for your platform, build from source: https://github.com/open-quantum-safe/liboqs-python

Datasets (PIMA, Cleveland, WDBC) are downloaded automatically on first run from public UCI/GitHub sources.


## Usage

```bash
# PQC only, 5 nodes, 30 rounds
~/pqc-env/bin/python simulation.py --rounds 30 --nodes 5 --mode pqc

# Head-to-head PQC vs Classical comparison
~/pqc-env/bin/python simulation.py --rounds 30 --mode both

# Multi-seed evaluation (5 seeds, 50 rounds each)
~/pqc-env/bin/python run_multiseed.py
```

| Flag | Default | Description |
|------|---------|-------------|
| `--rounds` | 5 | FL rounds |
| `--nodes` | 5 | Hospital nodes |
| `--mode` | `pqc` | `pqc`, `classical`, or `both` |
| `--lr` | 0.001 | Learning rate |
| `--seed` | 42 | Random seed |

Results are saved to a timestamped JSON file each run.


## Project Structure

```
├── simulation.py      # Entry point — orchestrates nodes, server, and metrics
├── node.py            # FederatedNode: local training + PQC/classical encrypt/sign
├── server.py          # AggregationServer: decrypt/verify + FedAvg aggregation
├── pqc_layer.py       # ML-KEM-768, ML-DSA-65, AES-256-GCM
├── classical_layer.py # RSA-4096 + HMAC-SHA256 baseline
├── data_utils.py      # Dataset loading, non-IID partitioning (5 hospitals)
├── run_multiseed.py   # Multi-seed evaluation script
└── results_multiseed.json  # 5-seed × 50-round benchmark output
```


## Planned Next Steps

- Byzantine fault demo: forged signature rejection (ML-DSA catches it) + gradient norm anomaly detection for poisoned-but-valid updates
- Security level comparison: ML-KEM-512/768/1024 × ML-DSA-44/65/87 vs latency
- Differential privacy: Gaussian noise on gradients before signing, epsilon-accuracy tradeoff curve
- Communication overhead in bytes: PQC vs classical bandwidth per round
- Scalability experiments: 5, 10, 20 hospital nodes


## References

- [NIST FIPS 203 — ML-KEM](https://csrc.nist.gov/pubs/fips/203/final)
- [NIST FIPS 204 — ML-DSA](https://csrc.nist.gov/pubs/fips/204/final)
- [Open Quantum Safe / liboqs](https://openquantumsafe.org/)
- McMahan et al., *Communication-Efficient Learning of Deep Networks from Decentralized Data* (FedAvg, 2017)


MIT License
