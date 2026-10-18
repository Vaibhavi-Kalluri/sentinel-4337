# Sentinel 4337 - Project Progress

## Project
Sentinel 4337: Zero Trust Threat Detection in ERC-4337 Alt Mempools via ZK-Gated Account Abstraction

## Environment

- [x] Kali Linux WSL2
- [x] Foundry 1.8.3
- [x] Forge
- [x] Anvil
- [x] Cast
- [x] Solidity compilation
- [x] Local Anvil blockchain
- [x] Chain ID 31337

## Smart Contract Infrastructure

- [x] SentinelTarget contract created
- [x] SentinelTarget compiled
- [x] ERC-4337 account-abstraction dependency installed
- [x] OpenZeppelin dependency installed
- [x] EntryPoint deployment script created
- [x] EntryPoint deployed locally
- [x] SimpleAccountFactory deployed locally
- [x] EntryPoint bytecode verified
- [x] SimpleAccountFactory bytecode verified

## Current ERC-4337 Deployment

EntryPoint:
0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512

SimpleAccountFactory:
0x9fE46736679d2D9a65F0992F2272dE9f3c7fa6e0

Chain:
Anvil local network

Chain ID:
31337

RPC:
http://127.0.0.1:8545

## Next Implementation Stages

- [ ] Deploy/create Smart Account
- [ ] Generate ERC-4337 UserOperations
- [ ] Execute UserOperations through EntryPoint
- [ ] Collect execution traces
- [ ] Convert traces into execution graphs
- [ ] Generate graph features
- [ ] Generate wallet behavior features
- [ ] Generate temporal/UserOperation context
- [ ] Build dataset
- [ ] Train baseline ML models
- [ ] Evaluate metadata-only model
- [ ] Evaluate metadata + wallet model
- [ ] Evaluate metadata + graph model
- [ ] Evaluate full feature fusion
- [ ] Implement GNN
- [ ] Implement temporal/context model
- [ ] Build risk engine
- [ ] Implement ZK gate
- [ ] Integrate complete pipeline
- [ ] Evaluate complete system

## Important

Anvil is a local development blockchain. Restarting Anvil normally resets its blockchain state.

The deployment scripts and recorded addresses are therefore kept in this repository so the environment can be reproduced.
