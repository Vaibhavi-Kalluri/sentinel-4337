<<<<<<< HEAD
## Foundry

**Foundry is a blazing fast, portable and modular toolkit for Ethereum application development written in Rust.**

Foundry consists of:

- **Forge**: Ethereum testing framework (like Truffle, Hardhat and DappTools).
- **Cast**: Swiss army knife for interacting with EVM smart contracts, sending transactions and getting chain data.
- **Anvil**: Local Ethereum node, akin to Ganache, Hardhat Network.
- **Chisel**: Fast, utilitarian, and verbose solidity REPL.

## Documentation

https://book.getfoundry.sh/

## Usage

### Build

```shell
$ forge build
```

### Test

```shell
$ forge test
```

### Format

```shell
$ forge fmt
```

### Gas Snapshots

```shell
$ forge snapshot
```

### Anvil

```shell
$ anvil
```

### Deploy

```shell
$ forge script script/Counter.s.sol:CounterScript --rpc-url <your_rpc_url> --private-key <your_private_key>
```

### Cast

```shell
$ cast <subcommand>
```

### Help

```shell
$ forge --help
$ anvil --help
$ cast --help
```
=======
\# Sentinel 4337



\## Zero-Trust Threat Detection in ERC-4337 Alt Mempools via ZK-Gated Account Abstraction



Sentinel 4337 is a research prototype that adds a behavioral and risk-adaptive security layer to ERC-4337 account abstraction.



The system aims to detect suspicious ERC-4337 UserOperations before execution using:



\- Rule-based threat detection

\- Machine-learning anomaly detection

\- Zero-Trust risk policies

\- Zero-knowledge proof-based authorization for high-risk operations



\## Problem



ERC-4337 provides protocol-level validation and account abstraction, but a cryptographically valid UserOperation is not necessarily benign.



Sentinel explores whether behavioral analysis can identify anomalous or potentially malicious UserOperations before execution.



\## Proposed Pipeline



UserOperation  

↓  

Alt Mempool  

↓  

Sentinel Monitoring  

↓  

Feature Extraction  

↓  

Rules + ML Detection  

↓  

Risk Score  

↓  

Zero-Trust Policy  

↓  

ZK Gate for High-Risk Operations  

↓  

Execute / Block



\## Status



🚧 Currently under development.



\## Technology Stack



\- Ethereum

\- ERC-4337

\- Solidity

\- Python

\- Machine Learning

\- Zero-Knowledge Proofs

\- Docker

\- Git/GitHub



\## Research Goals



1\. Detect anomalous ERC-4337 UserOperations.

2\. Compare rule-based detection with machine-learning detection.

3\. Apply Zero-Trust risk-based authorization.

4\. Explore ZK-gated authorization for high-risk operations.

5\. Measure detection performance and system overhead.



\## Project Structure



```text

sentinel-4337/

├── blockchain/

├── data/

├── detection/

├── zerotrust/

├── zk/

├── experiments/

├── results/

└── docs/

>>>>>>> origin/main
