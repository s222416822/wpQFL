import numpy as np
from config import num_devices, n_class, alpha

def create_noniid_data(x_train, y_train, num_devices, n_class, alpha, noniid_type):
    """
    Create non-IID data partitions for federated learning.

    Args:
        x_train: Training data features.
        y_train: Training data labels.
        num_devices: Number of devices.
        n_class: Number of classes per device for partial_class method.
        alpha: Dirichlet distribution parameter.
        noniid_type: Type of non-IID partitioning ('iid', 'partial_class', etc.).

    Returns:
        list: List of (x, y) tuples for each device.
    """

    def dirichlet_partitioning(x_data, y_data, num_devices, alpha=0.5):
        class_data = [x_data[y_data == i] for i in np.unique(y_data)]
        class_labels = [y_data[y_data == i] for i in np.unique(y_data)]
        device_data, device_labels = [[] for _ in range(num_devices)], [[] for _ in range(num_devices)]

        for c_data, c_labels in zip(class_data, class_labels):
            proportions = np.random.dirichlet([alpha] * num_devices)
            class_splits = np.split(c_data, (np.cumsum(proportions[:-1]) * len(c_data)).astype(int))
            label_splits = np.split(c_labels, (np.cumsum(proportions[:-1]) * len(c_labels)).astype(int))
            for i, (split_data, split_labels) in enumerate(zip(class_splits, label_splits)):
                device_data[i].append(split_data)
                device_labels[i].append(split_labels)

        device_data = [np.concatenate(d) for d in device_data]
        device_labels = [np.concatenate(l) for l in device_labels]
        return device_data, device_labels

    def enhanced_dirichlet_partitioning(x_data, y_data, num_devices, alpha=0.1, min_samples_per_class=5):
        class_data = [x_data[y_data == i] for i in np.unique(y_data)]
        class_labels = [y_data[y_data == i] for i in np.unique(y_data)]
        device_data, device_labels = [[] for _ in range(num_devices)], [[] for _ in range(num_devices)]

        for c_data, c_labels in zip(class_data, class_labels):
            proportions = np.random.dirichlet([alpha] * num_devices)
            proportions = np.maximum(proportions, min_samples_per_class / len(c_data))
            proportions /= proportions.sum()
            class_splits = np.split(c_data, (np.cumsum(proportions[:-1]) * len(c_data)).astype(int))
            label_splits = np.split(c_labels, (np.cumsum(proportions[:-1]) * len(c_labels)).astype(int))
            for i, (split_data, split_labels) in enumerate(zip(class_splits, label_splits)):
                device_data[i].append(split_data)
                device_labels[i].append(split_labels)

        device_data = [np.concatenate(d) for d in device_data]
        device_labels = [np.concatenate(l) for l in device_labels]
        return device_data, device_labels

    if noniid_type == "iid":
        indices = np.arange(len(x_train))
        np.random.shuffle(indices)
        split_indices = np.array_split(indices, num_devices)
        noniid_data = [(x_train[idx], y_train[idx]) for idx in split_indices]

    elif noniid_type == "partial_class":
        noniid_data = []
        for i in range(num_devices):
            classes = [(i + j) % 10 for j in range(n_class)]
            device_x, device_y = [], []
            for c in classes:
                device_x.append(x_train[y_train == c])
                device_y.append(y_train[y_train == c])
            device_x = np.concatenate(device_x)
            device_y = np.concatenate(device_y)
            noniid_data.append((device_x, device_y))

    elif noniid_type == "class_imbalance":
        noniid_data = []
        for _ in range(num_devices):
            proportions = np.random.dirichlet(np.ones(10), size=1).flatten()
            device_x, device_y = [], []
            for i, p in enumerate(proportions):
                class_data = x_train[y_train == i]
                class_labels = y_train[y_train == i]
                class_size = int(p * len(class_data))
                device_x.append(class_data[:class_size])
                device_y.append(class_labels[:class_size])
            device_x = np.concatenate(device_x)
            device_y = np.concatenate(device_y)
            noniid_data.append((device_x, device_y))

    elif noniid_type == "dirichlet":
        device_data, device_labels = dirichlet_partitioning(x_train, y_train, num_devices, alpha)
        noniid_data = list(zip(device_data, device_labels))

    elif noniid_type == "enhanced_dirichlet":
        device_data, device_labels = enhanced_dirichlet_partitioning(x_train, y_train, num_devices, alpha,
                                                                     min_samples_per_class=10)
        noniid_data = list(zip(device_data, device_labels))

    return noniid_data