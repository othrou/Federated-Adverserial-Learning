# Adversarial Machine Learning in Federated Learning (FL)

We created this repository as the end of module project, the module named 'Federated Learning'.

This repository is a practical implementation of **Adversarial Machine Learning in a Federated Context**, exploring the intersection of robustness, privacy, and decentralized learning. 

## Overview
Federated Learning (FL) [[1, 2, 3, 4]](#references) has emerged as a pivotal machine learning framework, addressing key challenges in data privacy, security, and decentralized data processing. Since its introduction by Google in 2017, FL has gained significant traction due to its ability to overcome major limitations in traditional centralized approaches. 

This project implements **adversarial attacks and defenses** in FL, reproducing foundational papers while adhering to a modular and consistent coding style. The goal is to build a well-structured library that integrates multiple FL strategies and adversarial techniques for robust benchmarking.

---

## Key Features
- **Modular FL Framework**: Implements FedAvg and other FL strategies with a standardized structure.
- **Adversarial ML Integration**: Supports adversarial attacks (e.g., model poisoning, data poisoning) and defenses (e.g., robust aggregation, differential privacy).
- **Benchmarking**: Comparative evaluation on MNIST, Fashion-MNIST, and CIFAR-10 under homogeneous and heterogeneous data splits.
- **Reproducibility**: Fixed experimental setup for fair comparisons.

---

## Repository Structure
All contributions adhere to the following package structure:





---

## Experimentation
### Datasets
- MNIST
- Fashion-MNIST (FMNIST)
- CIFAR-10

### Setup
- **Rounds**: 20 (or 40 if hardware permits)
- **Model**: Simple CNN (fixed architecture)
- **Clients**: 10 clients with 10 local epochs each

### Data Partitioning
1. **Homogeneous**: Even distribution across clients.
2. **Label Quantity (#C = k)**: Each client gets data from exactly `k` classes (tested for `k = 1, 2, 3`).
3. **Dirichlet**: Sample proportions from `Dir(β)` distribution.

---

## Getting Started
1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
