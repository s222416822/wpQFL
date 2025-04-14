import pennylane as qml
import torch
from torch.autograd import Variable
import numpy as np
from config import num_qubits, num_layers, num_classes, margin

dev = qml.device("default.qubit", wires=num_qubits)

def layer(W):
    """
    Apply a layer of rotations and CNOT gates.
    """
    for i in range(num_qubits):
        qml.Rot(W[i, 0], W[i, 1], W[i, 2], wires=i)
    for j in range(num_qubits - 1):
        qml.CNOT(wires=[j, j + 1])
    if num_qubits >= 2:
        # Apply additional CNOT to entangle the last with the first qubit
        qml.CNOT(wires=[num_qubits - 1, 0])

# @qml.qnode(device=dev, interface="torch")
def circuit(weights, feat=None):
    """
    Quantum circuit using amplitude embedding.
    """
    qml.AmplitudeEmbedding(feat, range(num_qubits), pad_with=0.0, normalize=True)
    for W in weights:
        layer(W)
    return qml.expval(qml.PauliZ(0))

qnodes = [qml.QNode(circuit, dev, interface="torch") for _ in range(num_classes)]

def variational_classifier(q_circuit, params, feat):
    """
    Variational classifier with bias.
    """
    weights = params[0]
    bias = params[1]
    return q_circuit(weights, feat=feat) + bias


# @qml.qnode(device=dev, interface="autograd")
# def circuit(weights, angles):
#   statepreparation(angles)
#   for W in weights:
#       layer(W)
#   circ = qml.expval(qml.PauliZ(0))
#   return circ

# def square_loss(labels, predictions):
#     loss = 0
#     for l, p in zip(labels, predictions):
#         loss = loss + (l - p) ** 2
#     loss = loss / len(labels)
#     return loss


def multiclass_svm_loss(q_circuits, all_params, feature_vecs, true_labels):
    """
    Compute multiclass SVM loss.
    """
    loss = 0
    num_samples = len(true_labels)
    for i, feature_vec in enumerate(feature_vecs):
        s_true = variational_classifier(
            q_circuits[int(true_labels[i])],
            (all_params[0][int(true_labels[i])], all_params[1][int(true_labels[i])]),
            feature_vec
        )
        s_true = s_true.float()
        li = 0
        for j in range(num_classes):
            if j != int(true_labels[i]):
                s_j = variational_classifier(
                    q_circuits[j], (all_params[0][j], all_params[1][j]), feature_vec
                )
                s_j = s_j.float()
                li += torch.max(torch.zeros(1).float(), s_j - s_true + margin)
        loss += li
    return loss / num_samples

def classify(q_circuits, all_params, feature_vecs, labels):
    """
    Classify samples using multiclass SVM.
    """
    predicted_labels = []
    for i, feature_vec in enumerate(feature_vecs):
        scores = np.zeros(num_classes)
        for c in range(num_classes):
            score = variational_classifier(
                q_circuits[c], (all_params[0][c], all_params[1][c]), feature_vec
            )
            scores[c] = float(score)
        pred_class = np.argmax(scores)
        predicted_labels.append(pred_class)
    return predicted_labels

def accuracy(labels, predictions):
    """
    Compute accuracy.
    """
    loss = 0
    for l, p in zip(labels, predictions):
        if torch.abs(l - p) < 1e-5:
            loss += 1
    return loss / labels.shape[0]


# def cost(weights, bias, X, Y):
#     # Transpose the batch of input data in order to make the indexing
#     # in state_preparation work
#     predictions = variational_classifier(weights, bias, X.T)
#     return square_loss(Y, predictions)

def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])


def state_preparation(a):
    qml.RY(a[0], wires=0)

    qml.CNOT(wires=[0, 1])
    qml.RY(a[1], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[2], wires=1)

    qml.PauliX(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[3], wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RY(a[4], wires=1)
    qml.PauliX(wires=0)

#References:
#https://github.com/PennyLaneAI/qml