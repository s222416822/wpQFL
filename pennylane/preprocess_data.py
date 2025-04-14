import torch
import numpy as np
from sklearn.datasets import load_iris
from config import num_devices, num_classes, train_split, feature_size


def preprocess_data(noniid_type="iid", alpha=0.1):
    """
    Load and preprocess Iris dataset with non-IID partitioning.

    Args:
        noniid_type (str): 'iid', 'single_class', 'partial_class', 'class_imbalance', 'dirichlet'.
        alpha (float): Dirichlet distribution parameter.

    Returns:
        tuple: (device_data, server_train_data, server_test_data)
    """
    # Load dataset
    try:
        data = np.loadtxt("iris.csv", delimiter=",")
    except FileNotFoundError:
        print("iris.csv not found, using scikit-learn Iris dataset.")
        iris = load_iris()
        data = np.column_stack((iris.data, iris.target))

    X = torch.tensor(data[:, 0:feature_size], dtype=torch.float32)
    normalization = torch.sqrt(torch.sum(X ** 2, dim=1))
    X_norm = X / normalization.reshape(len(X), 1)
    Y = torch.tensor(data[:, -1], dtype=torch.int64)

    # Split into train and test
    num_data = len(Y)
    num_train = int(train_split * num_data)
    index = np.random.permutation(range(num_data))
    X_train = X_norm[index[:num_train]]
    y_train = Y[index[:num_train]]
    X_test = X_norm[index[num_train:]]
    y_test = Y[index[num_train:]]

    # Combine the splits back into a single tensor for train and test sets
    train_data = torch.cat((X_train, y_train.unsqueeze(1)), dim=1)
    server_test_data = torch.cat((X_test, y_test.unsqueeze(1)), dim=1)

    # Split server data
    X_server = server_test_data[:, :-1]
    y_server = server_test_data[:, -1]
    num_server_data = len(y_server)
    num_server_train = int(train_split * num_server_data)
    server_index = np.random.permutation(range(num_server_data))
    X_server_train = X_server[server_index[:num_server_train]]
    y_server_train = y_server[server_index[:num_server_train]]
    X_server_test = X_server[server_index[num_server_train:]]
    y_server_test = y_server[server_index[num_server_train:]]

    server_train_data = torch.cat((X_server_train, y_server_train.unsqueeze(1)), dim=1)
    server_test_data = torch.cat((X_server_test, y_server_test.unsqueeze(1)), dim=1)

    # Non-IID partitioning
    if noniid_type == "iid":
        train_data = train_data[torch.randperm(train_data.size(0))]
        device_data = torch.chunk(train_data, num_devices)

    # Single-class data setup (Extreme Non-IID)

    elif noniid_type == "single_class":
        device_data = [train_data[train_data[:, -1] == i] for i in range(num_devices)]


    # Partial-class data setup (Moderate Non-IID)

    elif noniid_type == "partial_class":
        device_data = [
            torch.cat((train_data[train_data[:, -1] == 0], train_data[train_data[:, -1] == 1])),
            torch.cat((train_data[train_data[:, -1] == 1], train_data[train_data[:, -1] == 2])),
            torch.cat((train_data[train_data[:, -1] == 0], train_data[train_data[:, -1] == 2]))
        ]


    # Class-imbalanced data setup (Low Non-IID)

    elif noniid_type == "class_imbalance":
        proportions = [
            [0.7, 0.2, 0.1],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6]
        ]
        device_data = []
        for prop in proportions:
            device_subset = []
            for i, p in enumerate(prop):
                class_data = train_data[train_data[:, -1] == i]
                class_size = int(p * len(class_data))
                device_subset.append(class_data[:class_size])
            device_data.append(torch.cat(device_subset))

    # Dirichlet Distribution-Based Assignment (Controlled Non-IID)

    elif noniid_type == "dirichlet":
        class_indices = [train_data[train_data[:, -1] == i] for i in torch.unique(train_data[:, -1])]
        device_data = [[] for _ in range(num_devices)]
        for class_data in class_indices:
            proportions = np.random.dirichlet([alpha] * num_devices)
            split_sizes = (torch.tensor(proportions) * len(class_data)).int().tolist()
            split_sizes[-1] = len(class_data) - sum(split_sizes[:-1])
            class_splits = torch.split(class_data, split_sizes)
            for i, split in enumerate(class_splits):
                device_data[i].append(split)
        device_data = [torch.cat(parts) if parts else torch.tensor([]) for parts in device_data]

    else:
        raise ValueError(f"Unknown noniid_type: {noniid_type}")

    print(f"\n{noniid_type.replace('_', ' ').title()} Distribution:")
    for i, d in enumerate(device_data):
        if len(d) > 0:
            label_distribution = torch.bincount(d[:, -1].to(torch.int64), minlength=num_classes)
            print(f"Device {i + 1} data size: {len(d)}, Label distribution: {label_distribution}")
        else:
            print(f"Device {i + 1} data size: 0, Label distribution: None")

    return device_data, server_train_data, server_test_data