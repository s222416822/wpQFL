import numpy as np
import torch

# Set random seeds
np.random.seed(0)
torch.manual_seed(0)

# Configuration parameters
num_devices = 3
num_classes = 3
feature_size = 4
margin = 0.15
num_layers = 6
batch_size = 10
train_split = 0.75
num_qubits = int(np.ceil(np.log2(feature_size)))
folder = "pennylane"
data_used = "iris_normal"