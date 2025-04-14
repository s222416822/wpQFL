import numpy as np
import matplotlib.pyplot as plt
from preprocess_data import preprocess_data
from partition_data import create_noniid_data
from device import generate_devices
from training import main_method
from config import num_devices, n_class, alpha, dataset_used


def plot_distribution(noniid_data, noniid_type):
    """
    Plot label distribution across devices.

    Args:
        noniid_data: List of (x, y) tuples for each device.
        noniid_type: Type of non-IID partitioning.
    """
    plt.figure(figsize=(15, 5))
    for i, (device_x, device_y) in enumerate(noniid_data):
        label_distribution = np.bincount(device_y, minlength=10)
        plt.bar(np.arange(10) + i * 0.1, label_distribution, width=0.1, label=f"Device {i + 1}")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.title(f"Label Distribution for {noniid_type.capitalize()} Assignment")
    plt.legend()
    plt.show()


def main():
    """
    Main function to run the federated learning experiment.
    """
    noniid_types = ["iid", "partial_class", "class_imbalance", "dirichlet", "enhanced_dirichlet"]
    methods = [
        # "Default",
        # "Weighted",
        "Euclidean"]
    distance_metrics = [
        "euclidean",
        #  "cosine",
        # "cityblock",
        # "minkowski",
        # "mahalanobis",
        # "chebyshev",
        # "braycurtis"
    ]

    x_train, y_train, x_test, y_test = preprocess_data(dataset=dataset_used)

    for noniid_type in noniid_types:
        print(f"\nNon-IID Type: {noniid_type}")
        noniid_data = create_noniid_data(x_train, y_train, num_devices, n_class, alpha, noniid_type)

        for i, (device_x, device_y) in enumerate(noniid_data):
            label_distribution = np.bincount(device_y, minlength=10)
            print(f"Device {i + 1} - Data size: {len(device_x)}, Label distribution: {label_distribution}")

        plot_distribution(noniid_data, noniid_type)
        devices_list = generate_devices(noniid_data)

        for method in methods:
            if method == "Default":
                g, l = 0, 0
                main_method(method, g, l, noniid_data, noniid_type, devices_list, x_test, y_test)
            elif method == "Weighted":
                g, l = 10, 90
                # for v in values:
                #   g = 10 + v
                #   l = 90 - v
                main_method(method, g, l, noniid_data, noniid_type, devices_list, x_test, y_test)
            elif method == "Euclidean":
                g, l = 0, 0
                for metric in distance_metrics:
                    print(f"Distance Metric: {metric}")
                    main_method(method, g, l, noniid_data, noniid_type, devices_list, x_test, y_test, metric)


if __name__ == "__main__":
    main()