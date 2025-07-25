import os
import pandas as pd
import joblib

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def compute_playstyle_clusters(df, feature_cols, classification):
    if classification == "College":
        kmeans_path = os.path.join(PROJECT_ROOT, "models", "kmeans_college.pkl")
    else:
        kmeans_path = os.path.join(PROJECT_ROOT, "models", "kmeans_noncollege.pkl")

    # Load model
    if not os.path.exists(kmeans_path):
        raise FileNotFoundError(f"❌ Cluster model not found at: {kmeans_path}")

    kmeans = joblib.load(kmeans_path)

    # Only use rows with complete features
    valid_rows = df.dropna(subset=feature_cols).copy()
    clusters = kmeans.predict(valid_rows[feature_cols])
    valid_rows["cluster"] = clusters

        

    # One-hot encode
    cluster_dummies = pd.get_dummies(valid_rows["cluster"], prefix="cluster")
    valid_rows = pd.concat([valid_rows, cluster_dummies], axis=1)

    # Merge back into original dataframe
    df = df.merge(valid_rows[["Name", "cluster"] + list(cluster_dummies.columns)], on="Name", how="left")
    df["cluster"] = df["cluster"].fillna(-1).astype(int)  # Optional: treat missing as cluster -1

    return df
