import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

### --- PLAYSTYLE CLUSTERING --- ###

def compute_playstyle_clusters(df, feature_cols, n_clusters=5, use_pca=True, pca_components=2, plot=True):
    """
    Performs PCA (optional) and KMeans clustering to assign playstyle labels.
    Adds 'Playstyle' column to the dataframe.
    """
    
    # Drop rows with missing required features
    df_clean = df.dropna(subset=feature_cols)
    X = df_clean[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Optional PCA
    if use_pca:
        pca = PCA(n_components=pca_components)
        X_scaled = pca.fit_transform(X_scaled)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)

    # Assign cluster labels back to dataframe
    df_clean = df_clean.copy()
    df_clean['Playstyle'] = clusters

    # Merge back with original dataframe to preserve all rows
    df = df.merge(df_clean[['Name', 'Playstyle']], on='Name', how='left')

    # Optional 2D Plot
    if plot and use_pca and pca_components == 2:
        plt.figure(figsize=(8,6))
        scatter = plt.scatter(X_scaled[:,0], X_scaled[:,1], c=clusters, cmap='viridis', alpha=0.7)
        plt.title("Playstyle Clusters (PCA 2D)")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.colorbar(scatter, label="Cluster")
        plt.show()

    return df
