"""Fuzzy document clustering with Gaussian Mixture Models.

GMM is useful here because document embeddings rarely belong to exactly one
topic. A post about NASA funding can be mostly about space while also touching
politics or technology. Unlike hard clustering, GMM returns a probability for
each cluster, so uncertainty is represented directly as a probability
distribution instead of being hidden behind one forced label.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from core.qdrant_store import QdrantDocumentStore


@dataclass(frozen=True)
class ClusteredDocument:
    """Document embedding and payload loaded from Qdrant."""

    id: int
    vector: list[float]
    text: str
    category: str


@dataclass(frozen=True)
class ClusterExperimentResult:
    """Model-selection metrics for one GMM configuration."""

    n_clusters: int
    bic: float
    aic: float
    silhouette: float | None


def load_document_embeddings(store: QdrantDocumentStore) -> list[ClusteredDocument]:
    #Load document ids, vectors, text, and category labels from Qdrant
    #qdrant = persistent semantci memory.  

    documents: list[ClusteredDocument] = []
    for point in store.load_all_points():
        payload = point.payload or {}
        if point.vector is None:
            continue

        documents.append(
            ClusteredDocument(
                id=int(payload.get("document_id", point.id)),
                vector=list(point.vector),
                text=str(payload.get("text", "")),
                category=str(payload.get("category", "")),
            )
        )

    return documents


def vectors_to_matrix(documents: list[ClusteredDocument]) -> np.ndarray:
    """Convert loaded Qdrant vectors into a 2D numpy matrix - ml models expect matrices"""

    if not documents:
        raise ValueError("No documents with vectors were loaded from Qdrant")

    return np.asarray([document.vector for document in documents], dtype=np.float32)


def reduce_for_clustering(      #PCA dimensionality reduction
    embeddings: np.ndarray,
    n_components: int = 25,
    random_state: int = 42,
) -> np.ndarray:
    """Compress embeddings before GMM training.

    Sentence-transformer embeddings are high-dimensional. With a modest number
    of documents, a GMM in the full 384-dimensional space can become
    overconfident and assign probability 1.0 almost everywhere. PCA keeps the
    strongest semantic directions while making uncertainty estimates more
    useful for fuzzy clustering.
    """

    max_components = min(embeddings.shape[0] - 1, embeddings.shape[1], n_components)
    if max_components < 2:
        return embeddings

    reducer = PCA(n_components=max_components, random_state=random_state)   #PCA compresses info
    return reducer.fit_transform(embeddings) #learning imp directions and prjects vectors into compresses space


def train_gmm(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42,
) -> GaussianMixture:
    """Train a soft clustering model over document embeddings."""

    model = GaussianMixture(    #clustering algo 
        n_components=n_clusters,
        covariance_type="diag",
        reg_covar=1e-3, #regularization 
        random_state=random_state,
        n_init=3, #trains model 3 times with diff rand starts
    )
    model.fit(embeddings)   #training step
    return model


def cluster_documents(  #master piprline function 
    store: QdrantDocumentStore,
    n_clusters: int = 20,
    pca_components: int = 25,
    random_state: int = 42,
) -> tuple[list[ClusteredDocument], GaussianMixture, np.ndarray]:
    """Train GMM and store dominant cluster plus probabilities in Qdrant."""

    documents = load_document_embeddings(store) #laods vectors from qdrant 
    embeddings = vectors_to_matrix(documents)   #convert to matrix
    clustering_features = reduce_for_clustering(    #compress vectors
        embeddings=embeddings,
        n_components=pca_components,
        random_state=random_state,
    )
    model = train_gmm(  #trins clustering model
        embeddings=clustering_features,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    probabilities = model.predict_proba(clustering_features)    #returns prob for each doc of which cluster it is a part of
    dominant_clusters = probabilities.argmax(axis=1)    #finds highest probability cluster

    for document, dominant_cluster, cluster_probabilities in zip(
        documents,
        dominant_clusters,
        probabilities,
    ):
        # Soft clustering matters because this full probability vector shows
        # whether a document is clearly assigned or semantically ambiguous.
        store.set_payload(      #adds cluster metadata back to vdb
            doc_id=document.id,
            payload={
                "dominant_cluster": int(dominant_cluster),
                "cluster_probabilities": [
                    float(probability) for probability in cluster_probabilities
                ],
            },
        )

    return documents, model, probabilities


def get_cluster_distribution(   #get stored cluster porbabilitsef or 1 document
    store: QdrantDocumentStore,
    doc_id: int,
) -> list[float] | None:
    """Return the stored cluster-probability distribution for one document."""

    point = store.get_point(doc_id) #fetches vector entry from qdrant 
    if point is None or point.payload is None:
        return None

    probabilities = point.payload.get("cluster_probabilities")  #gets stored clustering metadata- extract payload
    if probabilities is None:
        return None

    return [float(probability) for probability in probabilities]


def experiment_cluster_counts(  #find the best no of clusters for the given dataset
    embeddings: np.ndarray,
    cluster_counts: list[int],
    pca_components: int = 25,
    random_state: int = 42,
) -> list[ClusterExperimentResult]:
    """Compare GMM cluster counts with BIC, AIC, and silhouette score."""

    clustering_features = reduce_for_clustering(
        embeddings=embeddings,
        n_components=pca_components,
        random_state=random_state,
    )
    results: list[ClusterExperimentResult] = []
    for n_clusters in cluster_counts:
        model = train_gmm(
            embeddings=clustering_features,
            n_clusters=n_clusters,
            random_state=random_state,
        )
        labels = model.predict(clustering_features)

        silhouette = None
        if 1 < len(set(labels)) < len(clustering_features):
            silhouette = float(silhouette_score(clustering_features, labels))   # how cleanly sereperated are clusters

        results.append(
            ClusterExperimentResult(
                n_clusters=n_clusters,
                bic=float(model.bic(clustering_features)),  #accuracy vs completxity
                aic=float(model.aic(clustering_features)),  #god fit without overcomplicating - both need to be low
                silhouette=silhouette,
            )
        )

    return results
