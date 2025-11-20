import time
import os
from datetime import datetime
import torch
import numpy as np
from quantum_model import multiclass_svm_loss, classify, accuracy, qnodes
from distance_metrics import compute_distances
from config import batch_size, num_classes


def main_method(logs, avg_method, device_data, device_list, server_train_data, server_test_data, g=0, l=0,
                distance_metrics_used=None):

    if not os.path.exists(logs):
        os.makedirs(logs)

    X_server_train = server_train_data[:, :-1]
    y_server_train = server_train_data[:, -1].to(torch.int64)
    X_server_test = server_test_data[:, :-1]
    y_server_test = server_test_data[:, -1].to(torch.int64)

    for it in range(100):
        start_time = time.time_ns()
        cost_array = []
        acc_train_array = []
        acc_val_array = []

        weights_list = []
        biases_list = []

        for i, d in enumerate(device_list):
            batch_index = np.random.randint(0, d.num_train, (batch_size,))
            feats_train_batch = d.feats_train[batch_index]
            Y_train_batch = d.Y_train[batch_index]

            d.optimizer.zero_grad()
            curr_cost = multiclass_svm_loss(d.q_circuits, d.params, feats_train_batch, Y_train_batch)
            curr_cost.backward()
            d.optimizer.step()
            d.weights, d.biases = d.params

            predictions_train = classify(d.q_circuits, d.params, d.feats_train, d.Y_train)
            predictions_val = classify(d.q_circuits, d.params, d.feats_val, d.y_val)
            acc_train = accuracy(d.Y_train, predictions_train)
            acc_val = accuracy(d.y_val, predictions_val)
            cost_value = multiclass_svm_loss(d.q_circuits, d.params, d.features, d.Y)

            cost_array.append(curr_cost.item())
            acc_train_array.append(acc_train)
            acc_val_array.append(acc_val)

            print(
                f"Comm: {it + 1:5d} | Device: {i:5d} | Cost: {curr_cost.item():0.7f} | Acc train: {acc_train:0.7f} | Acc val: {acc_val:0.7f}"
            )
            with open(f"{logs}/local_train.txt", "a") as file:
                file.write(
                    f"Comm: {it + 1} - Device {i} - cost: {curr_cost.item():0.7f} - train_acc: {acc_train:0.7f} - val_acc: {acc_val:0.7f}\n")

            weights_list.append(d.weights)
            biases_list.append(d.biases)

        average_weights = [
            torch.mean(torch.stack([w[i] for w in weights_list]), dim=0)
            for i in range(num_classes)
        ]
        average_biases = [
            torch.mean(torch.stack([b[i] for b in biases_list]), dim=0)
            for i in range(num_classes)
        ]
        average_params = (average_weights, average_biases)

        predictions_val_server = classify(qnodes, average_params, X_server_train, y_server_train)
        predictions_test_server = classify(qnodes, average_params, X_server_test, y_server_test)
        acc_val_server = accuracy(y_server_train, predictions_val_server)
        acc_test_server = accuracy(y_server_test, predictions_test_server)
        curr_cost_server = multiclass_svm_loss(qnodes, average_params, X_server_train, y_server_train)

        print(
            f"Server - Comm: {it + 1:5d} | Val Acc: {acc_val_server:0.7f} | Test Acc: {acc_test_server:0.7f} | Cost: {curr_cost_server.item():0.7f}"
        )
        with open(f"{logs}/server.txt", "a") as file:
            file.write(
                f"Comm: {it + 1} - val_acc: {acc_val_server:0.7f} - test_acc: {acc_test_server:0.7f} - cost: {curr_cost_server.item():0.7f}\n")

        total_time = time.time_ns() - start_time
        with open(f"{logs}/comm_time.txt", "a") as file:
            file.write(f"Comm: {it + 1} - Time: {total_time}\n")

        for d in device_list:
            if avg_method == "Default":
                d.weights = [w.clone().requires_grad_(True) for w in average_weights]
                d.biases = [b.clone().requires_grad_(True) for b in average_biases]
                d.params = (d.weights, d.biases)

            elif avg_method == "Weighted":
                d.weights = [
                    ((g * avg_w + l * dev_w) / (g + l)).clone().requires_grad_(True)
                    for avg_w, dev_w in zip(average_weights, d.weights)
                ]
                d.biases = [
                    ((g * avg_b + l * dev_b) / (g + l)).clone().requires_grad_(True)
                    for avg_b, dev_b in zip(average_biases, d.biases)
                ]
                d.params = (d.weights, d.biases)

            elif avg_method == "Euclidean":
                ews, ewi = compute_distances(d.params, average_params, (d.old_weights, d.biases), distance_metrics_used)
                d.old_weights = [w.clone() for w in d.weights]
                if ews < ewi:
                    d.weights = [w.clone().requires_grad_(True) for w in average_weights]
                    d.biases = [b.clone().requires_grad_(True) for b in average_biases]
                else:
                    d.weights = [
                        ((avg_w + dev_w) / 2).clone().requires_grad_(True)
                        for avg_w, dev_w in zip(average_weights, d.weights)
                    ]
                    d.biases = [
                        ((avg_b + dev_b) / 2).clone().requires_grad_(True)
                        for avg_b, dev_b in zip(average_biases, d.biases)
                    ]
                d.params = (d.weights, d.biases)