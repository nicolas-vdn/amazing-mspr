from typing import Any

import pandas as pd
import numpy as np
from pandas.core.interchange.dataframe_protocol import DataFrame
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

    print(pca_dataset.head)

    return pca_dataset

def search_hyperparameters(pca_dataset: pd.DataFrame) -> int:
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

    # sensitivities = [0.5, 1]
    # eps_values = []
    # for S in sensitivities:
    S=0.5
    kneedle = KneeLocator(range(len(distances)), distances, S=S, curve='convex', direction='increasing')
    knee_idx = kneedle.knee
    if knee_idx is not None:
        eps = distances[knee_idx]
        print(f"eps détecté avec S={S} : {eps:.4f}")
        # eps_values.append(eps)
    else:
        eps = None
        print(f"Aucun coude détecté avec S={S}")
        # eps_values.append(None)

    # return eps_values
    return eps

def train_model(pca_dataset: pd.DataFrame, eps_values) -> Any:
    """Trains the clustering model.

    Args:
        pca_dataset: Training data of independent features.
        eps_values: Hyperparameters
    Returns:
        Trained model.
    """

    # for eps in eps_values:
    clustering = DBSCAN(eps=eps_values, min_samples=1000).fit(pca_dataset)
        # models.append(clustering)

    return clustering


def evaluate_model(model: Any, pca_dataset: pd.DataFrame, eps_value, seed: int) -> DataFrame:
    """Calculates and logs the coefficient of determination.

    Args:
        models: List of clustering models.
        scaled_dataset: Training data of independent features.
        eps_value: Hyperparameters
        seed: Random seed.
    Returns:
        Dictionary containing coefficient of determination.
    """
    max_sample_size = 1000
    rng = np.random.default_rng(seed=seed)

    labels = model.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = list(labels).count(-1) / len(labels)
    print(f"eps={eps_value:.2f} -> clusters: {n_clusters}, bruit: {noise_ratio * 100:.3f}%")

    mask = labels != -1
    X_no_noise = pca_dataset[mask]
    labels_no_noise = labels[mask]

    sil_score = None
    ch_score = None
    db_score = None

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

    else:
        print("Pas assez de clusters pour calculer les scores.")


    evaluation_results = pd.DataFrame({"sil_score": [sil_score], "ch_score": [ch_score], "db_score": [db_score]})

    return evaluation_results
