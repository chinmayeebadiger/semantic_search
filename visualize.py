"""Dimensionality reduction and cluster visualization utilities."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


def reduce_embeddings_pca(
    embeddings: np.ndarray,
    n_components: int = 2,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce embeddings for visualization with PCA.

    PCA gives a fast, deterministic projection that helps inspect whether GMM
    clusters occupy distinct regions. UMAP can be added later if the project
    needs nonlinear structure, but PCA keeps Part 2 lightweight.
    """

    reducer = PCA(n_components=n_components, random_state=random_state)
    return reducer.fit_transform(embeddings)


def plot_clusters(
    reduced_embeddings: np.ndarray,
    dominant_clusters: np.ndarray,
    probabilities: np.ndarray,
    output_path: str | Path = "cluster_visualization.png",
) -> Path:
    """Create a 2D cluster plot colored by dominant cluster.

    Point opacity represents uncertainty: high-confidence documents appear more
    solid, while ambiguous documents are lighter because probability mass is
    spread across multiple clusters.
    """

    path = Path(output_path)
    confidence = probabilities.max(axis=1)
    alpha_values = np.clip(confidence, 0.25, 1.0)

    plt.figure(figsize=(11, 8))
    scatter = plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c=dominant_clusters,
        cmap="tab20",
        alpha=alpha_values,
        s=28,
        edgecolors="none",
    )
    plt.colorbar(scatter, label="Dominant cluster")
    plt.title("GMM Fuzzy Clusters of 20 Newsgroups Embeddings")
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def plot_model_selection(
    cluster_counts: list[int],
    bic_scores: list[float],
    aic_scores: list[float],
    silhouette_scores: list[float | None],
    output_path: str | Path = "cluster_model_selection.png",
) -> Path:
    """Plot BIC, AIC, and silhouette score for cluster-count experiments."""

    path = Path(output_path)
    valid_silhouette = [
        score if score is not None else np.nan for score in silhouette_scores
    ]

    fig, first_axis = plt.subplots(figsize=(10, 6))
    first_axis.plot(cluster_counts, bic_scores, marker="o", label="BIC")
    first_axis.plot(cluster_counts, aic_scores, marker="o", label="AIC")
    first_axis.set_xlabel("Number of clusters")
    first_axis.set_ylabel("Information criterion")
    first_axis.legend(loc="upper left")

    second_axis = first_axis.twinx()
    second_axis.plot(
        cluster_counts,
        valid_silhouette,
        color="tab:green",
        marker="o",
        label="Silhouette",
    )
    second_axis.set_ylabel("Silhouette score")
    second_axis.legend(loc="upper right")

    plt.title("GMM Cluster Count Experiments")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close(fig)
    return path
