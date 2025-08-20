import logging
from typing import Any

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import max_error, mean_absolute_error, r2_score
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
    pca_dataset = pca.fit_transform(dataset)

    return pca_dataset

def search_hyperparameters(dataset: pd.DataFrame) -> tuple:
    """Splits data into features and targets training and test sets.

    Args:
        data: Data containing features and target.
        parameters: Parameters defined in parameters/data_science.yml.
    Returns:
        Split data.
    """

    min_samples = dataset.shape[1] + 1

    neighbors = NearestNeighbors(n_neighbors=min_samples)
    neighbors_fit = neighbors.fit(dataset)
    distances, indices = neighbors_fit.kneighbors(dataset)

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


def train_model(dataset: pd.DataFrame, eps_values) -> list[Any]:
    """Trains the clustering model.

    Args:
        dataset: Training data of independent features.
        eps_values: Hyperparameters
    Returns:
        Trained model.
    """
    models = []

    for eps in eps_values:
        clustering = DBSCAN(eps=eps, min_samples=1000).fit(dataset)
        models.append(clustering)

    return models


def evaluate_model(models: list[Any], dataset: pd.DataFrame, eps_values, seed: int) -> dict[str, Any]:
    """Calculates and logs the coefficient of determination.

    Args:
        models: List of clustering models.
        dataset: Training data of independent features.
        eps_values: Hyperparameters
        seed: Random seed.
    Returns:
        Dictionary containing coefficient of determination.
    """
    max_sample_size = 1000

    best_score = -1
    best_eps = None
    rng = np.random.default_rng(seed=seed)

    for model in models:
        labels = model.labels_

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_ratio = list(labels).count(-1) / len(labels)
        print(f"eps={eps_values[models.index(model)]:.2f} -> clusters: {n_clusters}, bruit: {noise_ratio * 100:.3f}%")

        mask = labels != -1
        X_no_noise = dataset[mask]
        labels_no_noise = labels[mask]

        if n_clusters > 1 and len(X_no_noise) > 0:
            sample_size = min(max_sample_size, len(X_no_noise))
            indices = rng.choice(len(X_no_noise), sample_size, replace=False)
            X_sample = X_no_noise[indices]
            labels_sample = labels_no_noise[indices]

            sil_score = silhouette_score(X_sample, labels_sample)
            ch_score = calinski_harabasz_score(X_sample, labels_sample)
            db_score = davies_bouldin_score(X_sample, labels_sample)

            print(f"  Silhouette Score    : {sil_score:.3f}")
            print(f"  Calinski-Harabasz   : {ch_score:.3f}")
            print(f"  Davies-Bouldin      : {db_score:.3f}")

            if sil_score > best_score:
                best_score = sil_score
                best_eps = eps_values[models.index(model)]

        else:
            print("Pas assez de clusters pour calculer les scores.")

        return {"best_eps": best_eps, "best_score": best_score}
