from scipy.spatial import distance

def compute_distances(device_params, avg_params, old_params, metric, cov_matrix=None):
    device_params_flat = device_params.ravel()
    avg_params_flat = avg_params.ravel()
    old_params_flat = old_params.ravel()

    if metric == "euclidean":
        ews = distance.euclidean(device_params_flat, avg_params_flat)
        ewi = distance.euclidean(device_params_flat, old_params_flat)
    elif metric == "manhattan" or metric == "cityblock":
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