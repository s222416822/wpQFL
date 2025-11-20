import os
import time
import random
import numpy as np
from sklearn.utils import resample
from datetime import datetime
from data_loading import load_data
from device import Device
from non_iid_data import prepare_noniid_data
from compute_distance import compute_distances
from qiskit_algorithms.utils import algorithm_globals

# Set environment variables
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

# Parameters
comm_round = 10
test_data_size_used_value = 50
data_used = "genomics"
data_size = "small"
num_devices = 3
random_number = 1
simulator = "fake_manila"
algorithm_globals.random_seed = 1234


def main_method(method, g, l, noniid_data, noniid_type, distance_metrics_used=None):
    if distance_metrics_used is None:
        distance_metrics_used = "Not Required"

    server_device = None
    devices_list = []
    features, labels = noniid_data

    # Initialize devices
    for i in range(num_devices):
        device = Device(idx=i, data=features[i], labels=labels[i], maxiter=random_number, warm_start=True)
        devices_list.append(device)

    # Prepare server test data
    server_test_features, server_test_labels = alldevices_train_features, alldevices_train_labels
    unique_labels = np.unique(server_test_labels)
    server_test_features_small = []
    server_test_labels_small = []

    for label in unique_labels:
        label_indices = np.where(server_test_labels == label)[0]
        label_sample_size = min(test_data_size_used_value // len(unique_labels), len(label_indices))
        sampled_indices = resample(label_indices, n_samples=label_sample_size, replace=False, random_state=42)
        server_test_features_small.extend(server_test_features[sampled_indices])
        server_test_labels_small.extend(server_test_labels[sampled_indices])

    server_test_features_small = np.array(server_test_features_small)
    server_test_labels_small = np.array(server_test_labels_small)

    server_device = Device(idx=num_devices, data=server_test_features_small, labels=server_test_labels_small,
                           maxiter=random_number, warm_start=True)

    # Setup logging
    date_time = datetime.now().strftime("%m%d%Y_%H%M%S")
    logs = f"logs"

    if not os.path.exists(logs):
        os.makedirs(logs)


    print(f"Logs written to {logs}/parameters.txt")

    if method == "default":
        for n in range(comm_round):
            average_weights = None
            comm_start_time = time.time()
            total_weights = []
            for device in devices_list:
                if n > 0:
                    device.vqc.initial_point = device.weights_p
                device.training()
                with open(f"{logs}/device_{device.idx}_params.txt", "a") as file:
                    file.write(f"comm_round: {n} - vqc_weights:{device.vqc.weights} - weights_p: {device.weights_p}\n")
                device.new_params = device.vqc.weights
                total_weights.append(device.vqc.weights)

                print(
                    f"Comm_round: {n} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}\n")
                with open(f"{logs}/device.txt", 'a') as file:
                    file.write(
                        f"Comm_round: {n} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}\n")
                with open(f"{logs}/training_time_device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")

            average_weights = np.mean(total_weights, axis=0)
            print("Server Average Weights: ", average_weights)


            server_device.evaluate(average_weights)
            print(
                f"Comm_round: {n} - Server Device Test: {server_device.idx} - test_acc: {server_device.test_score_q4_1}")
            with open(f"{logs}/server_test.txt", 'a') as file:
                file.write(
                    f"Comm_round: {n} - Device: {server_device.idx} - test_acc: {server_device.test_score_q4_1}\n")

            for device in devices_list:
                device.weights_p = average_weights

            with open(f"{logs}/objective_values_devices.txt", 'a') as file:
                for device in devices_list:
                    file.write(f"Device {device.idx}: {device.objective_func_vals}\n")

            comm_end_time = time.time() - comm_start_time
            print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
            with open(f"{logs}/comm_time.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")

    else:
        for n in range(comm_round):
            average_weights = None
            comm_start_time = time.time()
            total_weights = []
            for device in devices_list:
                if n > 0:
                    device.vqc.initial_point = device.weights_p
                device.training()
                with open(f"{logs}/device_{device.idx}_params.txt", "a") as file:
                    file.write(f"comm_round: {n} - vqc_weights:{device.vqc.weights} - weights_p: {device.weights_p}\n")
                device.new_params = device.vqc.weights
                print("Device VQC weights", device.vqc.weights)
                total_weights.append(device.vqc.weights)

                print(
                    f"Comm_round: {n} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}\n")
                with open(f"{logs}/device.txt", 'a') as file:
                    file.write(
                        f"Comm_round: {n} - Device: {device.idx} - train_acc: {device.train_score_q4:.2f} - test_acc: {device.test_score_q4:.2f} - loss: {device.objective_func_vals[-1]}\n")
                with open(f"{logs}/training_time_device.txt", 'a') as file:
                    file.write(f"Comm_round: {n} - Device: {device.idx} - training_time: {device.training_time}\n")

            average_weights = np.mean(total_weights, axis=0)
            print("Average Weights: ", average_weights)


            server_device.evaluate(average_weights)

            if method == "weighted":
                for device in devices_list:
                    device.weights_p = (g * average_weights + l * device.new_params) / (g + l)

            if method == "euclidean":
                for device in devices_list:
                    ews, ewi = compute_distances(device.new_params, average_weights, device.old_params,
                                                 distance_metrics_used)
                    device.old_params = device.new_params
                    ed_avg = ews
                    ed_l = ewi
                    print("ED_AVG", ed_avg)
                    print("ED_L", ed_l)
                    print("EWS - EWI", ews, ewi)
                    if ed_avg < ed_l:
                        device.weights_p = average_weights
                    else:
                        device.weights_p = (average_weights + device.new_params) / 2

            with open(f"{logs}/objective_values_devices.txt", 'a') as file:
                for device in devices_list:
                    file.write(f"Device {device.idx}: {device.objective_func_vals}\n")


            comm_end_time = time.time() - comm_start_time
            print(f"Comm_round: {n} - Comm_time: {comm_end_time}")
            with open(f"{logs}/comm_time.txt", 'a') as file:
                file.write(f"Comm_round: {n} - Comm_time: {comm_end_time}\n")


if __name__ == "__main__":
    # Load data
    alldevices_train_features, server_test_features, alldevices_train_labels, server_test_labels = load_data(data_used,
                                                                                                             data_size)

    # Prepare non-IID data
    noniid_data_splits = prepare_noniid_data(alldevices_train_features, alldevices_train_labels, num_devices, data_size)

    # Run main method
    device_data = noniid_data_splits["enhanced_dirichlet"]
    main_method("euclidean", 0, 0, device_data, "enhanced_dirichlet", "euclidean")