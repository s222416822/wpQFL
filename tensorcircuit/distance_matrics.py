from scipy.spatial import distance


def compute_distances(device_params, avg_params, old_params, metric, cov_matrix=None):
    """
    Compute distances between parameter sets.

    Args:
        device_params: Current device parameters.
        avg_params: Average (global) parameters.
        old_params: Previous device parameters.
        metric: Distance metric to use.
        cov_matrix: Covariance matrix (optional).

    Returns:
        Tuple of distances (ews, ewi).
    """
    print("Metric Used:", metric)
    ews, ewi = 0, 0
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

    return ews, ewi