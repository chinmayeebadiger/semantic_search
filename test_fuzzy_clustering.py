"""Smoke test for Part 2 fuzzy clustering.

Run Qdrant first:
    docker compose up -d

Then make sure vectors exist:
    python test_semantic_search.py

Run:
    python test_fuzzy_clustering.py
"""

from __future__ import annotations

import numpy as np

from cluster_analysis import (
    ambiguous_documents,
    high_confidence_documents,
    print_representative_documents_per_cluster,
    print_scored_documents,
)
from gmm_cluster import (
    cluster_documents,
    experiment_cluster_counts,
    get_cluster_distribution,
    vectors_to_matrix,
)
from qdrant_store import QdrantDocumentStore
from visualize import plot_clusters, plot_model_selection, reduce_embeddings_pca


def main() -> None:
    """Train GMM, store cluster payloads, analyze results, and save plots."""

    store = QdrantDocumentStore(mode="localhost")

    documents, model, probabilities = cluster_documents(
        store=store,
        n_clusters=20,
    )
    embeddings = vectors_to_matrix(documents)
    dominant_clusters = probabilities.argmax(axis=1)

    print(f"Clustered {len(documents)} documents.")
    print(f"GMM converged: {model.converged_}")

    sample_doc_id = documents[0].id
    print(
        f"\nStored distribution for doc_id={sample_doc_id}: "
        f"{get_cluster_distribution(store, sample_doc_id)}"
    )

    print_representative_documents_per_cluster(
        documents=documents,
        probabilities=probabilities,
        top_n=2,
    )

    print_scored_documents(
        title="Ambiguous documents",
        documents=ambiguous_documents(documents, probabilities, top_n=5),
    )
    print_scored_documents(
        title="High-confidence documents",
        documents=high_confidence_documents(documents, probabilities, top_n=5),
    )

    cluster_counts = [5, 10, 15, 20, 25]
    experiments = experiment_cluster_counts(
        embeddings=embeddings,
        cluster_counts=cluster_counts,
    )
    print("\nCluster count experiments")
    for result in experiments:
        silhouette = (
            f"{result.silhouette:.4f}"
            if result.silhouette is not None
            else "n/a"
        )
        print(
            f"k={result.n_clusters} "
            f"BIC={result.bic:.2f} "
            f"AIC={result.aic:.2f} "
            f"silhouette={silhouette}"
        )

    reduced_embeddings = reduce_embeddings_pca(embeddings)
    cluster_plot = plot_clusters(
        reduced_embeddings=reduced_embeddings,
        dominant_clusters=dominant_clusters,
        probabilities=probabilities,
    )
    selection_plot = plot_model_selection(
        cluster_counts=[result.n_clusters for result in experiments],
        bic_scores=[result.bic for result in experiments],
        aic_scores=[result.aic for result in experiments],
        silhouette_scores=[result.silhouette for result in experiments],
    )

    cluster_sizes = np.bincount(dominant_clusters, minlength=20)
    print(f"\nCluster sizes: {cluster_sizes.tolist()}")
    print(f"Saved cluster plot: {cluster_plot}")
    print(f"Saved model-selection plot: {selection_plot}")


if __name__ == "__main__":
    main()
