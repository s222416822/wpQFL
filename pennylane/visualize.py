import matplotlib.pyplot as plt
from config import num_classes, num_devices
import torch


def plot_label_distribution(device_data, noniid_type):
    """
    Plot label distribution across devices.
    """
    plt.figure(figsize=(10, 6))
    plt.title(f"{noniid_type.replace('_', ' ').title()} Distribution")
    for i in range(num_devices):
        if len(device_data[i]) > 0:
            label_distribution = torch.bincount(device_data[i][:, -1].to(torch.int64), minlength=num_classes)
            plt.bar(range(num_classes) + i * 0.1, label_distribution, width=0.1, label=f"Device {i+1}")
    plt.xlabel("Labels")
    plt.ylabel("Frequency")
    plt.xticks(range(num_classes), [f"Class {i}" for i in range(num_classes)])
    plt.legend()
    plt.show()