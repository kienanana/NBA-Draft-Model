import numpy as np

# --- Feature Lists (must match features.py) ---
college_features = [
    "FGA", "FG%", "3PA", "3P%", "2PA", "2P%", "eFG%", "FTA", "FT%", "ORB", "TRB",
    "AST", "STL", "BLK", "TOV", "PF", "PTS", "PER", "TS%", "PProd", "ORB%", "DRB%", 
    "TRB%", "AST%", "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "OBPM", 
    "DBPM", "BPM", "EFG%", "3PAR", "FTAR", "NBA 3P%", "AST/USG", "AST/TO", "OWS/40", 
    "DWS/40", "ORTG", "DRTG"
]

noncollege_features = [
    "FGA", "FG%", "3PA", "3P%", "FTA", "FT%", "TRB", "AST", "STL", "BLK", "TOV", 
    "PF", "PTS", "PER", "TS%", "USG%", "EFG%", "3PAR", "FTAR", "NBA 3P%", 
    "AST/USG", "AST/TO", "ORTG", "DRTG"
]

# --- Helper Functions ---

def unpack_weights(weight_array, is_college=True):
    """
    Converts flat list of weights into structured offense/defense/general dict.
    """
    features = college_features if is_college else noncollege_features
    n = len(features)

    offense = dict(zip(features, weight_array[:n]))
    defense = dict(zip(features, weight_array[n:2*n]))
    general = dict(zip(features, weight_array[2*n:3*n]))

    return {"offense": offense, "defense": defense, "general": general}

def normalized_weight_array(is_college=True):
    """
    Generates a weight array where weights in each composite (off/def/gen) sum to 1.
    """
    features = college_features if is_college else noncollege_features
    n = len(features)

    # Equal weight per feature so that each category sums to 1
    one_third = np.ones(n) / n
    return np.concatenate([one_third, one_third, one_third])

def random_weight_array(is_college=True):
    """
    Generates a random weight array (values normalized per category).
    """
    features = college_features if is_college else noncollege_features
    n = len(features)

    weights = np.random.rand(n * 3).reshape(3, n)
    weights /= weights.sum(axis=1, keepdims=True)  # normalize each row to sum to 1
    return weights.flatten()

# --- Default Equal Weights for Development / Testing ---
college_weights = unpack_weights(normalized_weight_array(is_college=True), is_college=True)
noncollege_weights = unpack_weights(normalized_weight_array(is_college=False), is_college=False)
