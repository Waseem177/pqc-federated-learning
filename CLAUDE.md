# Project: PQC-Secured Federated Learning (Healthcare)

## What this is
5 simulated hospital nodes train locally on medical data.
Model updates are secured with CRYSTALS-Kyber (confidentiality) 
and ML-DSA/Dilithium (authentication) before aggregation.

## My role
PQC implementation and benchmarking.

## What's already working
- liboqs installed in ~/pqc-env
- Kyber512 key exchange working
- ML-DSA-44 signing and verification working
- AES-GCM payload encryption working
- Benchmark: PQC avg 0.411ms/round vs RSA 71ms/round

## Stack
- Python 3.12, Ubuntu WSL
- liboqs-python (ML-KEM = Kyber512, ML-DSA-44)
- cryptography library (AES-GCM)
- No Docker — simulated nodes in Python

## What needs building
1. 5 node simulation (each generates fake gradients)
2. PQC layer wrapping each node's update
3. FedAvg aggregation server with signature verification
4. Per-round latency and overhead metrics
5. Clean terminal dashboard

## Hard deadline
Saturday April 26, 5:30AM IST
