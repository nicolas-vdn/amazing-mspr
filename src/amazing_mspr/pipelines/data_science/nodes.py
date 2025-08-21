from typing import Any

import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from kneed import KneeLocator
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import DBSCAN


def pca_transformation(dataset: pd.DataFrame) -> pd.DataFrame:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """
    pca = PCA(n_components=3)
    pca_array = pca.fit_transform(dataset)
    pca_dataset = pd.DataFrame(pca_array)

    return pca_dataset

def search_hyperparameters(pca_dataset: pd.DataFrame) -> list[int]:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    min_samples = pca_dataset.shape[1] + 1

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(pca_dataset)
    distances, indices = neighbors_fit.kneighbors(pca_dataset)

    # On prend la distance au dernier voisin (le k-ième)
    distances = np.sort(distances[:, -1])

    sensitivities = [0.5, 1]
    eps_values = []
    for S in sensitivities:
        kneedle = KneeLocator(range(len(distances)), distances, S=S, curve='convex', direction='increasing')
        knee_idx = kneedle.knee
        if knee_idx is not None:
            eps = distances[knee_idx]
            print(f"eps détecté avec S={S} : {eps:.4f}")
            eps_values.append(eps)
        else:
            print(f"Aucun coude détecté avec S={S}")
            eps_values.append(None)

    return eps_values

def train_model(pca_dataset: pd.DataFrame, eps_values: list[int]) -> Any:
    """Trains the clustering model.

    Args:
        pca_dataset: Training data of independent features.
        eps_values: Hyperparameters
    Returns:
        Trained model.
    """

    max_sample_size = 1000
    min_samples = pca_dataset.shape[1] + 1
    rng = np.random.default_rng(seed=42)
    models = []

    best_sil_score = -1
    best_ch_score = None
    best_db_score = None
    best_eps = None

    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pca_dataset.to_numpy())
        models.append(clustering)
        labels = clustering.labels_
        print(labels)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = list(labels).count(-1) / len(labels)
        print(f"eps={eps:.2f} -> clusters: {n_clusters}, bruit: {noise_ratio * 100:.3f}%")

        mask = labels != -1
        X_no_noise = pca_dataset[mask].to_numpy()
        labels_no_noise = labels[mask]

        if n_clusters > 1 and len(X_no_noise) > 0:
            sample_size = min(max_sample_size, len(X_no_noise))
            indices = rng.choice(len(X_no_noise), sample_size, replace=False)
            X_sample = X_no_noise[indices]
            labels_sample = labels_no_noise[indices]
            unique_labels = np.unique(labels_sample)
            if len(unique_labels) > 1:
                sil_score = silhouette_score(X_sample, labels_sample)
                ch_score = calinski_harabasz_score(X_sample, labels_sample)
                db_score = davies_bouldin_score(X_sample, labels_sample)

                print(f"  Silhouette Score    : {sil_score:.3f}")
                print(f"  Calinski-Harabasz   : {ch_score:.3f}")
                print(f"  Davies-Bouldin      : {db_score:.3f}")

                if best_sil_score is None or sil_score > best_sil_score:
                    best_sil_score = sil_score
                    best_ch_score = ch_score
                    best_db_score = db_score
                    best_eps = eps
            else:
                print("L’échantillon n’a qu’un seul cluster, scores ignorés.")

        else:
            print("Pas assez de clusters pour calculer les scores.")

    # evaluation_results = pd.DataFrame(
    #     {"eps": [best_eps], "sil_score": [best_sil_score], "ch_score": [best_ch_score], "db_score": [best_db_score]})

    return models


# def evaluate_model(models: Any, pca_dataset: pd.DataFrame, eps_values, seed: int) -> DataFrame:
#     """Calculates and logs the coefficient of determination.
#
#     Args:
#         models: List of clustering models.
#         scaled_dataset: Training data of independent features.
#         eps_value: Hyperparameters
#         seed: Random seed.
#     Returns:
#         Dictionary containing coefficient of determination.
#     """
#     max_sample_size = 1000
#
#     best_eps = None
#     rng = np.random.default_rng(seed=seed)
#
#     best_sil_score = []
#     best_ch_score = []
#     best_db_score = []
#
#     for model in models:
#         current_eps = eps_values[models.index(model)]
#         labels = model.labels_
#
#         n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
#         noise_ratio = list(labels).count(-1) / len(labels)
#         print(f"eps={current_eps:.2f} -> clusters: {n_clusters}, bruit: {noise_ratio * 100:.3f}%")
#
#         mask = labels != -1
#         X_no_noise = pca_dataset[mask]
#         labels_no_noise = labels[mask]
#
#         if n_clusters > 1 and len(X_no_noise) > 0:
#             sample_size = min(max_sample_size, len(X_no_noise))
#             indices = rng.choice(len(X_no_noise), sample_size, replace=False)
#             X_sample = X_no_noise[indices]
#             labels_sample = labels_no_noise[indices]
#
#             sil_score = silhouette_score(X_sample, labels_sample)
#             ch_score = calinski_harabasz_score(X_sample, labels_sample)
#             db_score = davies_bouldin_score(X_sample, labels_sample)
#
#             print(f"  Silhouette Score    : {sil_score:.3f}")
#             print(f"  Calinski-Harabasz   : {ch_score:.3f}")
#             print(f"  Davies-Bouldin      : {db_score:.3f}")
#
#             if sil_score > best_sil_score:
#                 best_sil_score = sil_score
#                 best_ch_score = ch_score
#                 best_db_score = db_score
#                 best_eps = current_eps
#
#         else:
#             print("Pas assez de clusters pour calculer les scores.")
#
#
#     evaluation_results = pd.DataFrame({"eps": [best_eps], "sil_score": [best_sil_score], "ch_score": [best_ch_score], "db_score": [best_db_score]})
#
#     return evaluation_results
