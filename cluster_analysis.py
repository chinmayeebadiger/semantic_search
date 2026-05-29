"""Utilities for interpreting fuzzy GMM document clusters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gmm_cluster import ClusteredDocument


@dataclass(frozen=True)
class DocumentClusterScore:
    """A document plus its cluster-confidence summary."""

    doc_id: int
    dominant_cluster: int
    confidence: float
    category: str
    text: str


def _preview(text: str, max_length: int = 220) -> str:
    """Return a compact one-line document preview."""

    return text.replace("\n", " ")[:max_length]


def score_documents(
    documents: list[ClusteredDocument],
    probabilities: np.ndarray,
) -> list[DocumentClusterScore]:
    """Compute dominant cluster and confidence for each document."""

    scores: list[DocumentClusterScore] = []
    dominant_clusters = probabilities.argmax(axis=1)
    confidence_scores = probabilities.max(axis=1)

    for document, dominant_cluster, confidence in zip(
        documents,
        dominant_clusters,
        confidence_scores,
    ):
        scores.append(
            DocumentClusterScore(
                doc_id=document.id,
                dominant_cluster=int(dominant_cluster),
                confidence=float(confidence),
                category=document.category,
                text=document.text,
            )
        )

    return scores


def representative_documents_per_cluster(
    documents: list[ClusteredDocument],
    probabilities: np.ndarray,
    top_n: int = 3,
) -> dict[int, list[DocumentClusterScore]]:
    """Return high-probability representative documents for every cluster."""

    scored_documents = score_documents(documents, probabilities)
    representatives: dict[int, list[DocumentClusterScore]] = {}

    for cluster_id in range(probabilities.shape[1]):
        cluster_docs = [
            score
            for score in scored_documents
            if score.dominant_cluster == cluster_id
        ]
        representatives[cluster_id] = sorted(
            cluster_docs,
            key=lambda score: score.confidence,
            reverse=True,
        )[:top_n]

    return representatives


def ambiguous_documents(
    documents: list[ClusteredDocument],
    probabilities: np.ndarray,
    threshold: float = 0.55,
    top_n: int = 10,
) -> list[DocumentClusterScore]:
    """Find documents whose top cluster probability is low.

    Uncertainty is represented by probability mass being spread across several
    clusters. A low maximum probability means the model does not see a single
    dominant topic clearly.
    """

    scored_documents = score_documents(documents, probabilities)
    uncertain = [
        score for score in scored_documents if score.confidence <= threshold
    ]
    return sorted(uncertain, key=lambda score: score.confidence)[:top_n]


def high_confidence_documents(
    documents: list[ClusteredDocument],
    probabilities: np.ndarray,
    threshold: float = 0.85,
    top_n: int = 10,
) -> list[DocumentClusterScore]:
    """Find documents with one strongly dominant cluster."""

    scored_documents = score_documents(documents, probabilities)
    confident = [
        score for score in scored_documents if score.confidence >= threshold
    ]
    return sorted(confident, key=lambda score: score.confidence, reverse=True)[:top_n]


def print_representative_documents_per_cluster(
    documents: list[ClusteredDocument],
    probabilities: np.ndarray,
    top_n: int = 3,
) -> None:
    """Print representative document previews for each cluster."""

    representatives = representative_documents_per_cluster(
        documents=documents,
        probabilities=probabilities,
        top_n=top_n,
    )

    for cluster_id, cluster_documents in representatives.items():
        print(f"\nCluster {cluster_id}")
        for document in cluster_documents:
            print(
                f"  doc_id={document.doc_id} "
                f"confidence={document.confidence:.3f} "
                f"category={document.category}"
            )
            print(f"  {_preview(document.text)}")


def print_scored_documents(
    title: str,
    documents: list[DocumentClusterScore],
) -> None:
    """Print ambiguous or high-confidence document summaries."""

    print(f"\n{title}")
    for document in documents:
        print(
            f"doc_id={document.doc_id} "
            f"cluster={document.dominant_cluster} "
            f"confidence={document.confidence:.3f} "
            f"category={document.category}"
        )
        print(_preview(document.text))
