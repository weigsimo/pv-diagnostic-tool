import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import hdbscan
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import warnings


def create_shading_clusters(feature_vectors, out_dir=None, min_cluster_size=10, min_samples=5, verbose_output=False, store_results=True, create_plots=False):
    results = {}
    
    if store_results and out_dir:
        output_path = Path(out_dir) / "shading"
        output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    for plant_id, df in feature_vectors.items():
        if df.empty:
            continue
        
        plant_df = df.copy()
        plant_df["plant"] = plant_id
        all_data.append(plant_df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    non_feature_cols = ["date", "plant"]
    feature_cols = [col for col in combined_data.columns if col not in non_feature_cols]
    
    cluster_input = combined_data[feature_cols].dropna()
    valid_indices = cluster_input.index
    clustering_data = combined_data.loc[valid_indices].copy()
    
    if verbose_output:
        print(f"Clustering {len(clustering_data)} data points from {len(feature_vectors)} plants")
    
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='euclidean'
    )
    
    cluster_labels = clusterer.fit_predict(cluster_input)
    clustering_data["cluster"] = cluster_labels
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    if verbose_output:
        print(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")
    
    if create_plots:
        _create_clustering_plot(
            cluster_input.values, cluster_labels, clustering_data,
            "HDBSCAN Shading Clusters",
            output_path if store_results else None,
            "shading_clusters_pca.png"
        )
    
    for plant_id in feature_vectors.keys():
        plant_mask = clustering_data["plant"] == plant_id
        plant_results = clustering_data[plant_mask].copy()
        
        if not plant_results.empty:
            plant_results = plant_results.drop(columns=["plant"])
            results[plant_id] = plant_results
            
            if store_results and out_dir:
                output_file = output_path / f"{plant_id}_shading_clusters.csv"
                plant_results.to_csv(output_file, index=False)
    
    return results


def create_pollution_clusters(feature_vectors, out_dir=None, n_components=3, covariance_type="full", random_state=42, verbose_output=False, store_results=True, create_plots=False):
    results = {}
    
    if store_results and out_dir:
        output_path = Path(out_dir) / "pollution"
        output_path.mkdir(parents=True, exist_ok=True)
    
    all_data = []
    for plant_id, df in feature_vectors.items():
        if df.empty:
            continue
        
        plant_df = df.copy()
        plant_df["plant"] = plant_id
        all_data.append(plant_df)
    
    combined_data = pd.concat(all_data, ignore_index=True)
    
    non_feature_cols = ["date", "plant"]
    feature_cols = [col for col in combined_data.columns if col not in non_feature_cols]
    
    cluster_input = combined_data[feature_cols].dropna()
    valid_indices = cluster_input.index
    clustering_data = combined_data.loc[valid_indices].copy()
    
    if verbose_output:
        print(f"Clustering {len(clustering_data)} data points from {len(feature_vectors)} plants")
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(cluster_input)
    
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
        max_iter=200,
        n_init=10
    )
    
    cluster_labels = gmm.fit_predict(scaled_features)
    cluster_probabilities = gmm.predict_proba(scaled_features)
    
    clustering_data["cluster"] = cluster_labels
    
    for i in range(n_components):
        clustering_data[f"cluster_proba_{i}"] = cluster_probabilities[:, i]
    
    if verbose_output:
        print(f"GMM: {n_components} components")
    
    if create_plots:
        _create_clustering_plot(
            scaled_features, cluster_labels, clustering_data,
            "GMM Pollution Clusters",
            output_path if store_results else None,
            "pollution_clusters_pca.png"
        )
    
    for plant_id in feature_vectors.keys():
        plant_mask = clustering_data["plant"] == plant_id
        plant_results = clustering_data[plant_mask].copy()
        
        if not plant_results.empty:
            plant_results = plant_results.drop(columns=["plant"])
            results[plant_id] = plant_results
            
            if store_results and out_dir:
                output_file = output_path / f"{plant_id}_pollution_clusters.csv"
                plant_results.to_csv(output_file, index=False)
    
    return results


def _create_clustering_plot(features, labels, data, title, output_path=None, filename="clusters_pca.png"):
    pca = PCA(n_components=2, random_state=42)
    reduced_features = pca.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    
    scatter = plt.scatter(
        reduced_features[:, 0], 
        reduced_features[:, 1], 
        c=labels, 
        cmap="viridis", 
        alpha=0.7,
        s=50
    )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
    
    cbar = plt.colorbar(scatter)
    cbar.set_label("Cluster", fontsize=12)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plot_file = output_path / filename
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show() 
