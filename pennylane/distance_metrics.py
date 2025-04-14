from scipy.spatial import distance
import torch


def compute_distances(device_params, avg_params, old_params, metric, cov_matrix=None):
    """
    Compute distances between parameter sets.
    """
    ews, ewi = 0, 0
    # Convert the list of tensors to a single tensor
    # Flatten and concatenate all tensors in the lists to create a single 1D tensor

    device_params_flat = torch.cat([p.view(-1) for p in device_params[0] + device_params[1]])
    avg_params_flat = torch.cat([p.view(-1) for p in avg_params[0] + avg_params[1]])
    old_params_flat = torch.cat([p.view(-1) for p in old_params[0]])

    # Convert tensors to numpy arrays for distance calculations
    device_params_flat = device_params_flat.detach().cpu().numpy()
    avg_params_flat = avg_params_flat.detach().cpu().numpy()
    old_params_flat = old_params_flat.detach().cpu().numpy()

    # Calculate distance based on the chosen metric

    if metric == "euclidean":
        ews = distance.euclidean(device_params_flat, avg_params_flat)
        ewi = distance.euclidean(device_params_flat, old_params_flat)
    elif metric == "cityblock":
        ews = distance.cityblock(device_params_flat, avg_params_flat)
        ewi = distance.cityblock(device_params_flat, old_params_flat)
    elif metric == "minkowski":
        ews = distance.minkowski(device_params_flat, avg_params_flat, p=3)
        ewi = distance.minkowski(device_params_flat, old_params_flat, p=3)
    elif metric == "cosine":
        ews = distance.cosine(device_params_flat, avg_params_flat)
        ewi = distance.cosine(device_params_flat, old_params_flat)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return ews, ewi