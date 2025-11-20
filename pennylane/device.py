import torch
from torch.autograd import Variable
import torch.optim as optim
from quantum_model import qnodes, multiclass_svm_loss, classify, accuracy
from config import num_layers, num_qubits, num_classes
import pennylane as qml
from pennylane import numpy as np
from pennylane.optimize import NesterovMomentumOptimizer


class Device:
    def __init__(self, idx, feats_train, y_train, feats_val, y_val, num_train, features, Y):
        self.idx = idx
        self.dev = qml.device("default.qubit", wires=num_qubits)
        self.feats_train = feats_train
        self.Y_train = y_train
        self.feats_val = feats_val
        self.y_val = y_val
        self.num_train = num_train
        self.features = features
        self.Y = Y

        self.q_circuits = qnodes
        self.weights = [
            Variable(0.1 * torch.randn(num_layers, num_qubits, 3), requires_grad=True)
            for _ in range(num_classes)
        ]
        self.biases = [
            Variable(0.1 * torch.ones(1), requires_grad=True)
            for _ in range(num_classes)
        ]
        self.optimizer = optim.Adam(self.weights + self.biases, lr=0.01)
        self.params = (self.weights, self.biases)
        self.params_p = self.params
        self.old_weights = [w.clone() for w in self.weights]


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])

def initialize_devices(device_data):
    device_list = []
    for i, data in enumerate(device_data):
        opt = NesterovMomentumOptimizer(0.01)
        X = data[:, 0:2]
        padding = 0.3 * np.ones((len(X), 1))
        X_pad = np.c_[np.c_[X, padding], np.zeros((len(X), 1))]
        normalization = np.sqrt(np.sum(X_pad ** 2, -1))
        X_norm = (X_pad.T / normalization).T
        features = np.array([get_angles(x) for x in X_norm], requires_grad=False)
        Y = data[:, -1]

        np.random.seed(0)
        num_data1 = len(Y)
        num_train1 = int(0.75 * num_data1)
        index1 = np.random.permutation(range(num_data1))
        feats_train = features[index1[:num_train1]]
        Y_train = Y[index1[:num_train1]]
        feats_val = features[index1[num_train1:]]
        Y_val = Y[index1[num_train1:]]
        weights_init = 0.01 * np.random.randn(num_layers, num_qubits, 3, requires_grad=True)
        bias_init = np.array(0.0, requires_grad=True)
        device = Device(feats_train, Y_train, feats_val, Y_val, opt, weights_init, bias_init, num_train1, features, Y)
        device_list.append(device)

    return device_list