import numpy as np
from .features import college_features, noncollege_features

cluster_features = [f"cluster_{i}" for i in range(8)]

# --- Helper Functions ---

def unpack_weights(weight_array, is_college=True):
    """
    Converts flat list of weights into structured offense/defense/general/cluster dict.
    """
    features = college_features if is_college else noncollege_features
    n = len(features)

    offense = dict(zip(features, weight_array[:n]))
    defense = dict(zip(features, weight_array[n:2*n]))
    general = dict(zip(features, weight_array[2*n:3*n]))
    clusters = list(weight_array[3*n:3*n + 8])  # next 8 values

    return {
        "offense": offense,
        "defense": defense,
        "general": general,
        "clusters": clusters  # NEW
    }

def normalized_weight_array(is_college=True):
    features = college_features if is_college else noncollege_features
    n = len(features)

    one_third = np.ones(n) / n
    cluster_w = np.ones(8) / 8
    return np.concatenate([one_third, one_third, one_third, cluster_w])

def random_weight_array(is_college=True):
    features = college_features if is_college else noncollege_features
    n = len(features)

    weights = np.random.rand(n * 3).reshape(3, n)
    weights /= weights.sum(axis=1, keepdims=True)  # normalize offense/defense/general

    cluster_w = np.random.rand(8)
    cluster_w /= cluster_w.sum()  # normalize cluster weights

    return np.concatenate([weights.flatten(), cluster_w])

college_weights = unpack_weights(normalized_weight_array(is_college=True), is_college=True)
noncollege_weights = unpack_weights(normalized_weight_array(is_college=False), is_college=False)
