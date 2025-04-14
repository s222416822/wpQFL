# Performance Analysis and Design of a Weighted Personalized Quantum Federated Learning
IEEE Transactions on Artificial Intelligence
[https://10.1109/TAI.2025.3545393](URL) [Early Access]

This repository contains the code used for the paper titled "Performance Analysis and Design of a Weighted Personalized Quantum Federated Learning".

## Abstract 
Advances in federated and quantum computing have improved data privacy and efficiency in distributed systems. 
Quantum Federated Learning (QFL), like its classical counterpart, Classic Federated Learning (CFL), 
struggles with challenges in heterogeneous environments. 
To address these, we propose wp-QFL, a weighted personalized approach with quantum federated averaging (qFedAvg), 
tackling non-IID data and local model drift. While CFL personalization has been well explored, 
its application to QFL remains underdeveloped due to inherent differences. The proposed wp-QFL 
fills this gap by adapting to data heterogeneity with weighted personalization and drift correction. 


## Paper
- **Arxiv Preprint**: []()
- **Journal Version**: [https://10.1109/TAI.2025.3545393]() 

# Details
- Code Implementation for wpQFL Paper
- Includes implementation in Tensorcircuit, PennyLane and Qiskit
- Results in both simulators and Real IBM quantum Computer 

## Installation
- Qiskit, PennyLane, TensorCircuit
- For Qiskit:
  - pip install qiskit qiskit-algorithms qiskit-machine-learning genomic-benchmarks
- For PennyLane:
  - pip install pennylane
- For TensorCircuit
  - pip install tensorcircuit 
  - pip intall jax jaxlib
  - pip install optax
  - pip install cirq

## Folder Structure
- README.md file for general instructions and information
- wpQFL folder contains codes for each platform: Qiskit, TensorCircuit and PennyLane

References:
1. https://github.com/Qiskit/qiskit
2. Ville Bergholm et al. PennyLane: Automatic differentiation of hybrid quantum-classical computations. 2018. arXiv:1811.04968
2. Zhao, H. (2023). Non-IID quantum federated learning with one-shot communication complexity. Quantum Machine Intelligence, 5(1), 3.
2. https://tensorcircuit.readthedocs.io/en/latest/
3. https://github.com/PennyLaneAI/qml