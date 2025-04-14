import os

# Environment settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Configuration parameters
no_of_qubits = 10
no_of_classes = 10
n_node = 10
num_devices = n_node
k = 42
readout_mode = 'softmax'
dataset_used = "cifar"  # Can be changed to "mnist"
datasize_used = "normal"
n_class = 5
randClass = f"nClass={n_class}"
comm_rounds = 100
alpha = 0.1  # Dirichlet alpha parameter