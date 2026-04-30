"""
Similarity graphs and spectral geometry for auditory vocabularies.

Any collection of discrete auditory events — phonemes, notes, timbral
tokens, environmental sound classes — can be organised into a weighted
graph where edge weights encode pairwise similarity (acoustic, spectral,
perceptual, semantic, etc.).  The graph Laplacian of this structure
reveals clustering, connectivity, and the effective dimensionality of
the vocabulary.

This module provides the graph-construction and spectral-analysis
primitives.  The *similarity function* is supplied by the caller, so the
same pipeline works for:

- phonetic similarity (CMU pronunciation dictionary)
- spectral envelope similarity (MFCC distance)
- perceptual similarity (listener judgements)
- embedding similarity (any learned representation)

References
----------
Chung, F.R.K. (1997). Spectral Graph Theory. AMS.
Von Luxburg, U. (2007). "A tutorial on spectral clustering."
  Statistics and Computing.
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def build_similarity_graph(items, similarity_fn, threshold=0.1,
                           sample_pairs=None, seed=42):
    """
    Build a sparse, symmetric similarity graph over a vocabulary.

    Parameters
    ----------
    items : sequence of length N
        The vocabulary (words, tokens, sound events, …).
    similarity_fn : callable(a, b) → float
        Returns a similarity score in [0, 1] for two items.
        Must return 0.0 for identical items (no self-loops).
    threshold : float
        Minimum similarity to create an edge.
    sample_pairs : int or None
        If given, evaluate only this many random pairs instead of all
        N(N−1)/2.  Useful for large vocabularies.
    seed : int
        Random seed for pair sampling.

    Returns
    -------
    W : scipy.sparse.csr_matrix, shape (N, N)
        Symmetric adjacency / similarity matrix.
    """
    n = len(items)
    rows, cols, vals = [], [], []

    if sample_pairs is not None and sample_pairs < n * (n - 1) // 2:
        rng = np.random.RandomState(seed)
        for _ in range(sample_pairs):
            i, j = rng.randint(0, n, 2)
            if i == j:
                continue
            sim = similarity_fn(items[i], items[j])
            if sim > threshold:
                rows.extend([i, j])
                cols.extend([j, i])
                vals.extend([sim, sim])
    else:
        for i in range(n):
            for j in range(i + 1, n):
                sim = similarity_fn(items[i], items[j])
                if sim > threshold:
                    rows.extend([i, j])
                    cols.extend([j, i])
                    vals.extend([sim, sim])

    W = sparse.csr_matrix((vals, (rows, cols)), shape=(n, n))
    # Deduplicate by round-tripping through lil format
    W = W.tolil().tocsr()
    return W


def graph_laplacian(W):
    """
    Compute the combinatorial graph Laplacian L = D − W.

    Parameters
    ----------
    W : sparse matrix, shape (N, N)
        Symmetric adjacency / similarity matrix.

    Returns
    -------
    L : sparse matrix, shape (N, N)
        Graph Laplacian.
    degrees : ndarray, shape (N,)
        Degree of each node.
    """
    degrees = np.asarray(W.sum(axis=1)).flatten()
    D = sparse.diags(degrees)
    L = D - W
    return L, degrees


def spectral_analysis(L, k=20):
    """
    Compute the *k* smallest eigenvalues of a graph Laplacian.

    Parameters
    ----------
    L : sparse matrix, shape (N, N)
        Graph Laplacian (must be symmetric positive semi-definite).
    k : int
        Number of eigenvalues to compute.

    Returns
    -------
    eigenvalues : ndarray, shape (k,)
        Sorted non-negative eigenvalues.
    eigenvectors : ndarray, shape (N, k)
        Corresponding eigenvectors (columns).
    """
    n = L.shape[0]
    k = min(k, n - 2)
    if k < 2:
        return np.array([0.0]), np.ones((n, 1)) / np.sqrt(n)

    try:
        # Shift-invert for accurate small eigenvalues
        vals, vecs = eigsh(L.tocsc(), k=k, which="SM", sigma=1e-6)
    except Exception:
        try:
            # Fallback: dense solver for moderate sizes
            if n <= 2000:
                L_dense = L.toarray() if sparse.issparse(L) else L
                all_vals = np.linalg.eigvalsh(L_dense)
                all_vals = np.sort(np.maximum(all_vals, 0.0))
                return all_vals[:k], np.eye(n, k)
            vals, vecs = eigsh(L.tocsc(), k=k, which="SM")
        except Exception:
            return np.array([0.0]), np.ones((n, 1)) / np.sqrt(n)

    idx = np.argsort(vals)
    vals = np.maximum(vals[idx], 0.0)
    vecs = vecs[:, idx]
    return vals, vecs


def spectral_gap(eigenvalues):
    """
    Return the spectral gap λ₂ — the smallest *positive* eigenvalue.

    The spectral gap controls the mixing time of a random walk on the
    graph and measures how well-connected the vocabulary is.  A small
    gap means near-disconnected clusters; a large gap means tight
    global connectivity.

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues from :func:`spectral_analysis`.

    Returns
    -------
    float
        The spectral gap, or 0.0 if no positive eigenvalue exists.
    """
    positive = eigenvalues[eigenvalues > 1e-8]
    return float(positive[0]) if len(positive) > 0 else 0.0


def effective_dimension(eigenvalues, threshold=0.9):
    """
    Number of eigenvalues needed to capture *threshold* of total spectral
    weight.

    A low effective dimension means the graph's structure is well
    described by a few dominant modes (strong clustering).

    Parameters
    ----------
    eigenvalues : ndarray
        Sorted eigenvalues from :func:`spectral_analysis`.
    threshold : float
        Fraction of cumulative spectral weight to capture (default 0.9).

    Returns
    -------
    int
        Effective spectral dimension.
    """
    positive = eigenvalues[eigenvalues > 1e-8]
    if len(positive) == 0:
        return 0
    cumulative = np.cumsum(positive) / np.sum(positive)
    return int(np.searchsorted(cumulative, threshold)) + 1


def spectral_clustering(W, n_clusters, normalized=True, random_state=42):
    """
    Ng–Jordan–Weiss (2001) spectral clustering.

    Embed data points into the space spanned by the top eigenvectors of
    the (normalized) graph Laplacian, then run k‑means in that embedding.

    Parameters
    ----------
    W : sparse matrix, shape (N, N)
        Symmetric similarity matrix (affinity).
    n_clusters : int
        Number of clusters to find.
    normalized : bool
        If True, use the normalized Laplacian (Ng–Jordan–Weiss).
        If False, use the combinatorial Laplacian (Shi–Malik variant).
    random_state : int or None
        Seed for k‑means initialisation.

    Returns
    -------
    labels : ndarray, shape (N,)
        Cluster assignment (0 … n_clusters‑1).
    embedding : ndarray, shape (N, n_clusters)
        Eigenvector embedding (rows are points).
    """
    from scipy.sparse.linalg import eigsh
    from scipy.sparse import diags
    from sklearn.cluster import KMeans

    n = W.shape[0]
    if n_clusters >= n:
        return np.arange(n), np.ones((n, n_clusters)) / np.sqrt(n)

    if normalized:
        # NJW: work with the normalised affinity  D^{-1/2} W D^{-1/2}
        # and take its *largest* eigenvectors (equivalent to the smallest
        # eigenvectors of the normalised Laplacian L_sym = I − D^{-1/2} W D^{-1/2}).
        degrees = np.asarray(W.sum(axis=1)).flatten()
        sqrt_deg_inv = 1.0 / np.where(np.sqrt(degrees) > 0,
                                       np.sqrt(degrees), 1.0)
        D_inv_sqrt = diags(sqrt_deg_inv)
        A_norm = D_inv_sqrt @ W @ D_inv_sqrt

        # Dense path for moderate sizes — avoids shift-invert pitfalls
        if n <= 3000:
            A_dense = A_norm.toarray() if sparse.issparse(A_norm) else A_norm
            all_vals, all_vecs = np.linalg.eigh(A_dense)
            # Largest eigenvalues are at the end
            idx = np.argsort(all_vals)[::-1][:n_clusters]
            vals = all_vals[idx]
            vecs = all_vecs[:, idx]
        else:
            vals, vecs = eigsh(A_norm.tocsc(), k=n_clusters, which='LM')
    else:
        # Combinatorial Laplacian: L = D − W, smallest eigenvectors
        degrees = np.asarray(W.sum(axis=1)).flatten()
        D = diags(degrees)
        L = D - W
        if n <= 3000:
            L_dense = L.toarray() if sparse.issparse(L) else L
            all_vals, all_vecs = np.linalg.eigh(L_dense)
            idx = np.argsort(all_vals)[:n_clusters]
            vals = all_vals[idx]
            vecs = all_vecs[:, idx]
        else:
            vals, vecs = eigsh(L.tocsc(), k=n_clusters, which='SM')

    # Form embedding matrix X ∈ ℝ^{n × n_clusters}
    X = vecs
    # Row‑normalise X (Ng–Jordan–Weiss step)
    row_norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.where(row_norms > 0, row_norms, 1.0)

    # k‑means in the eigenvector space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state,
                    n_init=10)
    labels = kmeans.fit_predict(X)

    return labels, X
