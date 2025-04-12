import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns


def prepare_noniid_data(alldevices_train_features, alldevices_train_labels, num_devices, data_size):
    alpha = 0.1
    min_samples_per_class = 10
    noniid_data_splits = {}

    if data_size == "small":
        total_samples = 150
    else:
        total_samples = len(alldevices_train_features)

    random_indices = np.random.choice(len(alldevices_train_features), total_samples, replace=False)
    alldevices_train_features = alldevices_train_features[random_indices]
    alldevices_train_labels = alldevices_train_labels[random_indices]

    samples_per_device = len(alldevices_train_features) // num_devices
    remainder = len(alldevices_train_features) % num_devices

    # IID Data Split
    iid_data_features = []
    iid_data_labels = []
    start_index = 0
    for i in range(num_devices):
        extra_sample = 1 if i < remainder else 0
        end_index = start_index + samples_per_device + extra_sample
        iid_data_features.append(np.array(alldevices_train_features[start_index:end_index]))
        iid_data_labels.append(np.array(alldevices_train_labels[start_index:end_index]))
        start_index = end_index
    noniid_data_splits["iid"] = (iid_data_features, iid_data_labels)

    # Single-Class Data
    single_class_data_features = []
    single_class_data_labels = []
    unique_labels = np.unique(alldevices_train_labels)
    for i in range(num_devices):
        label = unique_labels[i % len(unique_labels)]
        indices = np.where(alldevices_train_labels == label)[0]
        single_class_data_features.append(np.array(alldevices_train_features[indices]))
        single_class_data_labels.append(np.array(alldevices_train_labels[indices]))
    noniid_data_splits["single_class"] = (single_class_data_features, single_class_data_labels)

    # Partial-Class Data
    partial_class_data_features = []
    partial_class_data_labels = []
    for i in range(num_devices):
        selected_labels = unique_labels[i:i + 2] if i < num_devices - 1 else unique_labels[:2]
        indices = np.where(np.isin(alldevices_train_labels, selected_labels))[0]
        partial_class_data_features.append(np.array(alldevices_train_features[indices]))
        partial_class_data_labels.append(np.array(alldevices_train_labels[indices]))
    noniid_data_splits["partial_class"] = (partial_class_data_features, partial_class_data_labels)

    # Class-Imbalance Data
    class_imbalance_data_features = []
    class_imbalance_data_labels = []
    for i in range(num_devices):
        proportion = np.random.dirichlet(np.ones(len(unique_labels)))
        device_features = []
        device_labels = []
        for j, label in enumerate(unique_labels):
            indices = np.where(alldevices_train_labels == label)[0]
            class_size = int(proportion[j] * len(indices))
            device_features.extend(alldevices_train_features[indices][:class_size])
            device_labels.extend(alldevices_train_labels[indices][:class_size])
        class_imbalance_data_features.append(np.array(device_features))
        class_imbalance_data_labels.append(np.array(device_labels))
    noniid_data_splits["class_imbalance"] = (class_imbalance_data_features, class_imbalance_data_labels)

    # Enhanced Dirichlet Data Split
    enhanced_dirichlet_data_features = [[] for _ in range(num_devices)]
    enhanced_dirichlet_data_labels = [[] for _ in range(num_devices)]
    for label in unique_labels:
        indices = np.where(alldevices_train_labels == label)[0]
        proportions = np.random.dirichlet([alpha] * num_devices)
        proportions = np.maximum(proportions, min_samples_per_class / len(indices))
        proportions /= proportions.sum()
        split_points = (proportions * len(indices)).astype(int).cumsum()
        last_idx = 0
        for i, point in enumerate(split_points):
            enhanced_dirichlet_data_features[i].extend(alldevices_train_features[indices][last_idx:point])
            enhanced_dirichlet_data_labels[i].extend(alldevices_train_labels[indices][last_idx:point])
            last_idx = point
        if last_idx < len(indices):
            enhanced_dirichlet_data_features[-1].extend(alldevices_train_features[indices][last_idx:])
            enhanced_dirichlet_data_labels[-1].extend(alldevices_train_labels[indices][last_idx:])

    enhanced_dirichlet_data_features = [np.array(d) for d in enhanced_dirichlet_data_features]
    enhanced_dirichlet_data_labels = [np.array(l) for l in enhanced_dirichlet_data_labels]
    noniid_data_splits["enhanced_dirichlet"] = (enhanced_dirichlet_data_features, enhanced_dirichlet_data_labels)

    # Debug and visualization
    for noniid_type, (features_list, labels_list) in noniid_data_splits.items():
        print(f"\nNon-IID Type: {noniid_type}")
        for i in range(num_devices):
            labels = labels_list[i]
            label_distribution = np.bincount(labels, minlength=len(unique_labels))
            print(f"Device {i + 1} data size: {len(features_list[i])}, Label distribution: {label_distribution}")

    for noniid_type, (features_list, labels_list) in noniid_data_splits.items():
        plt.figure(figsize=(10, 5))
        plt.title(f"Label Distribution for {noniid_type} Data Split")
        for i in range(num_devices):
            labels = labels_list[i]
            label_distribution = np.bincount(labels, minlength=len(unique_labels))
            plt.bar(range(len(unique_labels)), label_distribution, alpha=0.5, label=f"Device {i + 1}")
        plt.xlabel("Labels")
        plt.ylabel("Frequency")
        plt.xticks(range(len(unique_labels)), unique_labels)
        plt.legend()
        plt.show()

    return noniid_data_splits