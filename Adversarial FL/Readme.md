

## Overview

This project simulates federated learning (FL) with adversarial attacks to evaluate robustness and communication efficiency. It compares a standard CNN model with a communication-efficient variant called FedPara, which compresses model updates via low-rank Hadamard parameterization. Malicious clients apply Universal Adversarial Perturbations (UAP) targeting a specific class to study attack impact under different data partitioning scenarios.

In this directory, we performed some tests to show the vulnerabilities of FL systems agains adversarial attacks.


### Universal Adversarial Perturbation (UAP)
UAP is a powerful adversarial attack that creates a single perturbation pattern capable of fooling a model on **most input samples** when added to them. Key properties:

1. **Universal**: One perturbation works for many inputs (unlike input-specific adversarial examples)
2. **Transferable**: Often works across different models
3. **Stealthy**: Small perturbations are visually imperceptible

The **objective of UAP** is to find a perturbation vector $v$ that satisfies two key constraints:

1. The perturbation norm $\|v\|_p$ must be less than or equal to a threshold $\xi$, which controls the magnitude of the perturbation to keep it imperceptible or limited.

2. The probability that the classifier’s prediction changes under the perturbation is at least $1 - \delta$, where $\delta$ is a small tolerance parameter. Formally:

$$
\mathbb{P}_{x \sim \mu} \left( \hat{k}(x + v) \neq \hat{k}(x) \right) \geq 1 - \delta
$$

This means that the perturbation $v$ should cause misclassification on most inputs drawn from distribution $\mu$.


**In Federated Learning Context**:
- Malicious clients compute UAPs that cause misclassification of a target class
- They embed these perturbations in their model updates
- The global model eventually learns to misclassify the target class

---

### **Code Execution Step-by-Step**

#### **1. Setup Environment**
```bash

# Create conda environment (recommended)
conda create -n fl_uap python=3.8
conda activate fl_uap

# enter directory 
cd "UAP against FedPara"

# Install dependencies
pip install torch torchvision numpy matplotlib
or pip install -r requirements 
```

#### **2. Directory Structure**
```
/UAP against FedPara
│── utils.py          # Model definitions & attack implementations
│── strategy.py       # Data partitioning strategies  
│── client.py         # Client class
│── server.py         # Server class
│── visualizations.py # Plotting functions
└── simulation.py # Main script 
```

#### **3. Run the Experiment**
```bash
python simulation.py
```

#### **4. Key Execution Steps Explained**
1. **Initialization**:
   - Loads CIFAR-10 or any other data
   - Creates server and client instances
   - Malicious clients (first 3) initialize UAP attacks targeting class 0 ("plane")

2. **UAP Training Phase**:
   ```python
   # Inside Client initialization:
   if client.is_malicious:
       client.train_attack_if_needed(temp_model_for_attack_ref)
   ```
   - Malicious clients train their UAP generators using the initial global model

3. **Federated Rounds**:
   - Each round:
     1. Server broadcasts current model
     2. Clients perform local training (malicious ones apply UAP perturbations)
     3. Server aggregates updates via FedAvg/FedPara

4. **Evaluation**:
   - Tracks accuracy degradation on target class
   - Generates plots comparing different data partitioning strategies

---

### **Critical Attack Parameters**
```python
# In your configuration:
NUM_MALICIOUS_CLIENTS = 3          # Number of attacking clients
UAP_TARGET_CLASS_TO_PERTURB = 0    # Class index to attack (0=plane in CIFAR-10)
FEDPARA_RANK = 8                   # Compression rank for FedPara
```

---

### **Expected Output**
1. **Console Logs**:
   ```
   Using device: cuda
   Client 0 will be malicious, targeting class 0.
   Client 1 will be malicious, targeting class 0. 
   Client 2 will be malicious, targeting class 0.
   --- Round 1/20 (Homogeneous) ---
   [Evaluation] Accuracy: 45.2% | Target class acc: 12.3%
   ...
   Survival rate for target class plane under Homogeneous: 15.2%
   ```

2. **Generated Plots**:
   - `accuracy_vs_rounds.png`: Shows accuracy degradation
   - `class_accuracy_comparison.png`: Target class accuracy drop
   - `communication_savings.png`: FedPara vs Standard model size


## Results

![Alt text](https://github.com/yourusername/repo-name/blob/main/Adversarial%20FL/helper/global_accuracy.png?raw=true)
*Figure 1: Global model accuracy across federated learning rounds with UAP attacks*

---
