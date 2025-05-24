# FedDefenderZ: Server-Side Enhanced FedPara for Adversarially-Robust Federated Learning

FedDefenderZ is a server-side defense mechanism for Federated Learning (FL), designed to detect and mitigate adversarial client updates. It integrates three core techniques: anomaly detection using Isolation Forest, low-rank compression via FedPara, and robust aggregation through IRLS (Iteratively Reweighted Least Squares).

##  Features

-  Dynamic anomaly detection with Isolation Forest
-  Efficient compression using FedPara (low-rank factorization)
-  Robust aggregation via IRLS
-  Simulated label flipping and other poisoning attacks
-  Benchmarking across MNIST, FMNIST, and CIFAR-10
-  Support for multiple data partitioning strategies (IID, Dirichlet, Label-skewed)

---

##  Project Structure

```bash
├── simulation.py           # Main script to run experiments
├── strategy.py             # Implements FedParaIRLSStrategy
├── client.py               # Client class with attack simulation
├── server.py               # Server coordination logic
├── defence_utils.py        # IRLS and helper functions
├── utils.py                # Seed, saving, plotting utilities
├── models.py               # CNN models per dataset
├── dataset.py              # Dataset loading and partitioning
├── results/                # Auto-saved results and plots
└── README.md
