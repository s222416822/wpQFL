import torch
from datetime import datetime
from preprocess_data import preprocess_data
from device import initialize_devices
from training import main_method
from visualize import plot_label_distribution
from config import num_devices, folder, data_used
from pennylane import numpy as np


def get_angles(x):
    beta0 = 2 * np.arcsin(np.sqrt(x[1] ** 2) / np.sqrt(x[0] ** 2 + x[1] ** 2 + 1e-12))
    beta1 = 2 * np.arcsin(np.sqrt(x[3] ** 2) / np.sqrt(x[2] ** 2 + x[3] ** 2 + 1e-12))
    beta2 = 2 * np.arcsin(np.linalg.norm(x[2:]) / np.linalg.norm(x))

    return np.array([beta2, -beta1 / 2, beta1 / 2, -beta0 / 2, beta0 / 2])
      # Same as before

def main():
    """
    Main function to run the federated learning experiment.
    """
    noniid_types = ["iid", "single_class", "partial_class", "class_imbalance", "dirichlet"]
    avg_methods = [
        "Default",
        # "No AVG",
        # "LocalUpdate",
        # "Euclidean",
        # "Weighted",
        # "Euclidean_bias"
    ]

    # avg_method = "Default"
    # avg_method = "No AVG"
    # avg_method = "LocalUpdate"
    # avg_method = "Euclidean"
    # avg_method = "Weighted"
    # avg_method = "Euclidean_bias"

    distance_metrics = ["euclidean", "cosine", "cityblock", "minkowski"]
    values = [0, 10, 40, 70, 80]

    for noniid_type in noniid_types:
        print(f"\nNon-IID Type: {noniid_type}")
        device_data, server_train_data, server_test_data = preprocess_data(noniid_type=noniid_type)
        plot_label_distribution(device_data, noniid_type)
        device_list = initialize_devices(device_data)

        # Step 6: Prepare the server test data
        X_server = server_test_data[:, 0:2]
        padding = 0.3 * np.ones((len(X_server), 1))
        X_server_pad = np.c_[np.c_[X_server, padding], np.zeros((len(X_server), 1))]
        normalization_server = np.sqrt(np.sum(X_server_pad ** 2, -1))
        X_server_norm = (X_server_pad.T / normalization_server).T
        features_server = np.array([get_angles(x) for x in X_server_norm], requires_grad=False)
        Y_server = server_test_data[:, -1]

        # Step 7: Split the server test data into training and validation sets
        np.random.seed(0)
        num_data_server = len(Y_server)
        num_train_server = int(0.75 * num_data_server)
        index_server = np.random.permutation(range(num_data_server))
        feats_server_train = features_server[index_server[:num_train_server]]
        Y_server_train = Y_server[index_server[:num_train_server]]
        feats_server_val = features_server[index_server[num_train_server:]]
        Y_server_val = Y_server[index_server[num_train_server:]]

        for avg_method in avg_methods:
            date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
            if avg_method == "Default":
                logs = f"logs_{folder}_{data_used}/{date_time}_{avg_method}_devices{num_devices}_noniid_{noniid_type}"
                main_method(logs, avg_method, device_data, device_list, server_train_data, server_test_data)
            elif avg_method == "Weighted":
                for v in values:
                    g = 10 + v
                    l = 90 - v
                    logs = f"logs_{folder}_{data_used}/{date_time}_{avg_method}_devices{num_devices}_g{g}_l{l}_noniid_{noniid_type}"
                    main_method(logs, avg_method, device_data, device_list, server_train_data, server_test_data, g, l)
            elif avg_method == "Euclidean":
                for metric in distance_metrics:
                    logs = f"logs_{folder}_{data_used}/{date_time}_{avg_method}_devices{num_devices}_metric_{metric}_noniid_{noniid_type}"
                    main_method(logs, avg_method, device_data, device_list, server_train_data, server_test_data,
                                distance_metrics_used=metric)