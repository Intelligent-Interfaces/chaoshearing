"""
Reproducing the Analysis from Ng, Jordan & Weiss (2001)
=======================================================

"On Spectral Clustering: Analysis and an Algorithm"

This notebook reproduces the key experiments and figures from the NJW
paper, then extends them into the CASA (Computational Auditory Scene
Analysis) domain — showing spectral clustering as a principled mechanism
for auditory stream segregation.

Figures produced:
  1. Concentric circles — k-means fails, spectral clustering succeeds
  2. Interleaved half-moons — same story
  3. Eigenvector embedding — what the data looks like in Laplacian space
  4. Eigenvalue spectrum — the eigengap heuristic for choosing k
  5. CASA application — spectral clustering on a cochlear filterbank
     similarity graph, recovering auditory streams from a mixture

Usage:
    python notebooks/njw_spectral_clustering.py

Erick Oduniyi — Chaos Hearing / CASA
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles, make_moons

from casa.graphs import spectral_clustering, graph_laplacian, spectral_analysis
from casa.oscillators import cochlear_filterbank
from casa.kernels import spectral_mixture_kernel


# ── Colour palette ──────────────────────────────────────────────────────
C0 = "#E91E63"   # magenta-pink
C1 = "#2196F3"   # blue
C2 = "#4CAF50"   # green
C3 = "#FF9800"   # orange
C4 = "#9C27B0"   # purple
GREY = "#9E9E9E"
COLORS = [C0, C1, C2, C3, C4]


def gaussian_affinity(X, sigma):
    """Build a Gaussian (RBF) affinity matrix from point cloud X."""
    from scipy.spatial.distance import pdist, squareform
    D = squareform(pdist(X, "sqeuclidean"))
    W = np.exp(-D / (2.0 * sigma ** 2))
    np.fill_diagonal(W, 0.0)
    return sparse.csr_matrix(W)


# =====================================================================
# Figure 1 — Concentric Circles
# =====================================================================

def figure_concentric_circles():
    """
    NJW's motivating example: two concentric rings.
    k-means in ℝ² fails; spectral clustering succeeds.
    """
    np.random.seed(42)
    X, y_true = make_circles(n_samples=300, factor=0.4, noise=0.06)

    # k-means in original space
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_km = km.fit_predict(X)

    # Spectral clustering (NJW)
    W = gaussian_affinity(X, sigma=0.2)
    labels_sc, embedding = spectral_clustering(W, n_clusters=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("Figure 1 — Concentric Circles (NJW §4)", fontsize=13, y=1.02)

    # (a) Ground truth
    for c in [0, 1]:
        mask = y_true == c
        axes[0].scatter(X[mask, 0], X[mask, 1], s=12, c=COLORS[c],
                        alpha=0.7, edgecolors="none")
    axes[0].set_title("(a) Ground truth")
    axes[0].set_aspect("equal")

    # (b) k-means
    for c in [0, 1]:
        mask = labels_km == c
        axes[1].scatter(X[mask, 0], X[mask, 1], s=12, c=COLORS[c],
                        alpha=0.7, edgecolors="none")
    axes[1].set_title("(b) k-means in ℝ²")
    axes[1].set_aspect("equal")

    # (c) Spectral clustering
    for c in [0, 1]:
        mask = labels_sc == c
        axes[2].scatter(X[mask, 0], X[mask, 1], s=12, c=COLORS[c],
                        alpha=0.7, edgecolors="none")
    axes[2].set_title("(c) NJW spectral clustering")
    axes[2].set_aspect("equal")

    for ax in axes:
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    plt.tight_layout()
    return fig, X, W


# =====================================================================
# Figure 2 — Interleaved Half-Moons
# =====================================================================

def figure_half_moons():
    """Two interleaved half-moons — another classic non-convex case."""
    np.random.seed(7)
    X, y_true = make_moons(n_samples=300, noise=0.07)

    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    labels_km = km.fit_predict(X)

    W = gaussian_affinity(X, sigma=0.25)
    labels_sc, embedding = spectral_clustering(W, n_clusters=2)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("Figure 2 — Interleaved Half-Moons (NJW §4)", fontsize=13, y=1.02)

    for ax, labels, title in [
        (axes[0], y_true, "(a) Ground truth"),
        (axes[1], labels_km, "(b) k-means in ℝ²"),
        (axes[2], labels_sc, "(c) NJW spectral clustering"),
    ]:
        for c in [0, 1]:
            mask = labels == c
            ax.scatter(X[mask, 0], X[mask, 1], s=12, c=COLORS[c],
                       alpha=0.7, edgecolors="none")
        ax.set_title(title)
        ax.set_aspect("equal")
        ax.set_xlabel("x₁")
        ax.set_ylabel("x₂")

    plt.tight_layout()
    return fig


# =====================================================================
# Figure 3 — Eigenvector Embedding
# =====================================================================

def figure_eigenvector_embedding(X, W):
    """
    Show what the concentric circles look like after projecting onto
    the top-2 eigenvectors of the normalised Laplacian.
    """
    labels_sc, embedding = spectral_clustering(W, n_clusters=2)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle("Figure 3 — Eigenvector Embedding (NJW §3)",
                 fontsize=13, y=1.02)

    # (a) Original space
    for c in [0, 1]:
        mask = labels_sc == c
        axes[0].scatter(X[mask, 0], X[mask, 1], s=14, c=COLORS[c],
                        alpha=0.7, edgecolors="none")
    axes[0].set_title("(a) Original space ℝ²")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    axes[0].set_aspect("equal")

    # (b) Eigenvector space
    for c in [0, 1]:
        mask = labels_sc == c
        axes[1].scatter(embedding[mask, 0], embedding[mask, 1], s=14,
                        c=COLORS[c], alpha=0.7, edgecolors="none")
    axes[1].set_title("(b) Eigenvector space (row-normalised)")
    axes[1].set_xlabel("v₁ (1st eigenvector)")
    axes[1].set_ylabel("v₂ (2nd eigenvector)")
    axes[1].set_aspect("equal")

    # Annotate: the clusters are now linearly separable
    axes[1].annotate("linearly\nseparable",
                     xy=(0.5, 0.5), xycoords="axes fraction",
                     fontsize=9, color=GREY, ha="center",
                     fontstyle="italic")

    plt.tight_layout()
    return fig


# =====================================================================
# Figure 4 — Eigenvalue Spectrum & Eigengap Heuristic
# =====================================================================

def figure_eigenvalue_spectrum():
    """
    Build a 3-cluster dataset, compute the Laplacian eigenvalues,
    and show the eigengap that reveals k = 3.
    """
    np.random.seed(12)
    # Three Gaussian blobs with different densities
    n_per = 80
    blob_a = np.random.randn(n_per, 2) * 0.3 + np.array([-2, 0])
    blob_b = np.random.randn(n_per, 2) * 0.3 + np.array([2, 0])
    blob_c = np.random.randn(n_per, 2) * 0.3 + np.array([0, 2.5])
    X = np.vstack([blob_a, blob_b, blob_c])
    y_true = np.array([0]*n_per + [1]*n_per + [2]*n_per)

    W = gaussian_affinity(X, sigma=0.5)
    L, degrees = graph_laplacian(W)
    eigenvalues, eigenvectors = spectral_analysis(L, k=15)

    labels_sc, _ = spectral_clustering(W, n_clusters=3)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    fig.suptitle("Figure 4 — Eigenvalue Spectrum & Eigengap Heuristic",
                 fontsize=13, y=1.02)

    # (a) Data with spectral clustering labels
    for c in range(3):
        mask = labels_sc == c
        axes[0].scatter(X[mask, 0], X[mask, 1], s=14, c=COLORS[c],
                        alpha=0.7, edgecolors="none")
    axes[0].set_title("(a) 3 clusters — NJW result")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x₁")
    axes[0].set_ylabel("x₂")

    # (b) Eigenvalue spectrum
    k_plot = min(12, len(eigenvalues))
    axes[1].bar(range(k_plot), eigenvalues[:k_plot], color=GREY, alpha=0.7,
                edgecolor="black", linewidth=0.5)
    # Highlight the gap between λ₃ and λ₄
    if k_plot > 3:
        gap_val = eigenvalues[3] - eigenvalues[2]
        axes[1].annotate(
            f"eigengap\nΔλ = {gap_val:.3f}",
            xy=(3, eigenvalues[3]),
            xytext=(5, eigenvalues[3] + 0.02),
            fontsize=9, color=C0,
            arrowprops=dict(arrowstyle="->", color=C0, lw=1.2),
        )
        # Colour the first 3 bars
        for bar_idx in range(3):
            axes[1].patches[bar_idx].set_facecolor(C1)
            axes[1].patches[bar_idx].set_alpha(0.85)
    axes[1].set_xlabel("Eigenvalue index i")
    axes[1].set_ylabel("λᵢ")
    axes[1].set_title("(b) Laplacian eigenvalues")

    # (c) Consecutive eigenvalue differences (eigengap plot)
    diffs = np.diff(eigenvalues[:k_plot])
    axes[2].bar(range(1, len(diffs) + 1), diffs, color=GREY, alpha=0.7,
                edgecolor="black", linewidth=0.5)
    if len(diffs) > 2:
        max_gap_idx = np.argmax(diffs)
        axes[2].patches[max_gap_idx].set_facecolor(C0)
        axes[2].patches[max_gap_idx].set_alpha(0.9)
        axes[2].annotate(
            f"k = {max_gap_idx + 1}",
            xy=(max_gap_idx + 1, diffs[max_gap_idx]),
            xytext=(max_gap_idx + 2.5, diffs[max_gap_idx]),
            fontsize=10, color=C0, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=C0, lw=1.2),
        )
    axes[2].set_xlabel("Gap index (λᵢ₊₁ − λᵢ)")
    axes[2].set_ylabel("Δλ")
    axes[2].set_title("(c) Eigengap heuristic → k = 3")

    plt.tight_layout()
    return fig


# =====================================================================
# Figure 5 — CASA Application: Auditory Stream Segregation
# =====================================================================

def figure_casa_stream_segregation():
    """
    Spectral clustering as auditory stream segregation.

    Generate a synthetic mixture of two harmonic sources, run it through
    the cochlear filterbank, build a similarity graph over the channel
    responses, and use NJW spectral clustering to recover the two
    auditory streams.
    """
    np.random.seed(0)

    # Two harmonic sources
    sr = 16000
    duration = 0.1
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Source A: fundamental 220 Hz + harmonics (2f, 3f, 4f)
    source_a = (np.sin(2 * np.pi * 220 * t)
                + 0.6 * np.sin(2 * np.pi * 440 * t)
                + 0.3 * np.sin(2 * np.pi * 660 * t)
                + 0.15 * np.sin(2 * np.pi * 880 * t))

    # Source B: fundamental 330 Hz + harmonics
    source_b = (0.8 * np.sin(2 * np.pi * 330 * t)
                + 0.5 * np.sin(2 * np.pi * 660 * t)
                + 0.25 * np.sin(2 * np.pi * 990 * t))

    mixture = source_a + source_b + 0.02 * np.random.randn(len(t))

    # Cochlear filterbank
    n_channels = 48
    responses, freqs = cochlear_filterbank(
        mixture, sr, n_channels=n_channels,
        f_low=100, f_high=2000, mu=-0.05
    )

    # Build affinity matrix from channel response correlation
    # Channels whose envelopes are correlated likely belong to the same source
    # (common modulation — a Bregman grouping cue)
    envelope = responses  # amplitude envelopes
    # Normalise each channel
    norms = np.linalg.norm(envelope, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    envelope_norm = envelope / norms

    # Correlation-based affinity
    corr = envelope_norm @ envelope_norm.T  # (n_channels, n_channels)
    corr = np.maximum(corr, 0)  # keep positive correlations only
    np.fill_diagonal(corr, 0)
    W = sparse.csr_matrix(corr)

    # Spectral clustering → 2 streams
    labels, embedding = spectral_clustering(W, n_clusters=2)

    # Ground truth: channels near 220/440/880 Hz → source A,
    # channels near 330/660/990 Hz → source B
    source_a_freqs = {220, 440, 660, 880}
    source_b_freqs = {330, 660, 990}

    # ── Plotting ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        "Figure 5 — CASA: Spectral Clustering as Auditory Stream Segregation",
        fontsize=13, y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.4)

    # (a) Mixture waveform
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(t * 1000, mixture, linewidth=0.4, color="steelblue")
    ax.set_title("(a) Acoustic mixture")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")

    # (b) Individual sources
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t * 1000, source_a, linewidth=0.5, alpha=0.7, color=C0,
            label="Source A (f₀=220 Hz)")
    ax.plot(t * 1000, source_b, linewidth=0.5, alpha=0.7, color=C1,
            label="Source B (f₀=330 Hz)")
    ax.set_title("(b) Ground truth sources")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=7)

    # (c) Cochlear filterbank response (cochleagram)
    ax = fig.add_subplot(gs[0, 2])
    extent = [0, duration * 1000, freqs[0], freqs[-1]]
    ax.imshow(responses, aspect="auto", origin="lower", cmap="inferno",
              extent=extent, interpolation="nearest")
    ax.set_title("(c) Cochlear filterbank response")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")

    # (d) Affinity matrix
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(corr, cmap="YlOrRd", origin="lower", aspect="auto")
    ax.set_title("(d) Channel correlation (affinity W)")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Channel")
    plt.colorbar(im, ax=ax, fraction=0.046)

    # (e) Eigenvector embedding
    ax = fig.add_subplot(gs[1, 1])
    for c in range(2):
        mask = labels == c
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=30,
                   c=COLORS[c], alpha=0.8, edgecolors="black", linewidth=0.3,
                   label=f"Stream {c}")
    ax.set_title("(e) Eigenvector embedding")
    ax.set_xlabel("v₁")
    ax.set_ylabel("v₂")
    ax.legend(fontsize=8)

    # (f) Eigenvalue spectrum of the affinity Laplacian
    ax = fig.add_subplot(gs[1, 2])
    L, degrees = graph_laplacian(W)
    evals, _ = spectral_analysis(L, k=min(15, n_channels - 2))
    k_plot = min(12, len(evals))
    bars = ax.bar(range(k_plot), evals[:k_plot], color=GREY, alpha=0.7,
                  edgecolor="black", linewidth=0.5)
    if k_plot > 2:
        bars[0].set_facecolor(C1)
        bars[1].set_facecolor(C1)
        bars[0].set_alpha(0.85)
        bars[1].set_alpha(0.85)
        # Annotate eigengap
        if k_plot > 2:
            gap = evals[2] - evals[1]
            ax.annotate(f"eigengap\nΔλ = {gap:.2f}",
                        xy=(2, evals[2]),
                        xytext=(4, evals[2] + 0.5),
                        fontsize=8, color=C0,
                        arrowprops=dict(arrowstyle="->", color=C0, lw=1))
    ax.set_xlabel("Eigenvalue index")
    ax.set_ylabel("λᵢ")
    ax.set_title("(f) Laplacian eigenvalues")

    # (g) Cluster assignments mapped back to frequency
    ax = fig.add_subplot(gs[2, 0])
    for c in range(2):
        mask = labels == c
        ax.barh(np.where(mask)[0], freqs[mask], color=COLORS[c],
                alpha=0.8, height=0.8, label=f"Stream {c}")
    ax.set_xlabel("Characteristic frequency (Hz)")
    ax.set_ylabel("Channel index")
    ax.set_title("(g) Stream assignments by frequency")
    ax.legend(fontsize=8)

    # (h) Separated cochleagrams
    ax = fig.add_subplot(gs[2, 1])
    stream_0 = responses.copy()
    stream_0[labels != 0] = 0
    ax.imshow(stream_0, aspect="auto", origin="lower", cmap="Reds",
              extent=extent, interpolation="nearest", alpha=0.9)
    ax.set_title(f"(h) Stream 0 — {np.sum(labels==0)} channels")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")

    ax = fig.add_subplot(gs[2, 2])
    stream_1 = responses.copy()
    stream_1[labels != 1] = 0
    ax.imshow(stream_1, aspect="auto", origin="lower", cmap="Blues",
              extent=extent, interpolation="nearest", alpha=0.9)
    ax.set_title(f"(i) Stream 1 — {np.sum(labels==1)} channels")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Frequency (Hz)")

    plt.tight_layout(rect=[0, 0, 1, 0.96], w_pad=2, h_pad=2)
    return fig

def main():
    os.makedirs("notebooks", exist_ok=True)

    print("NJW Spectral Clustering — Reproducing the Analysis")
    print("=" * 52)

    # Figure 1
    print("\nFigure 1: Concentric circles...")
    fig1, X_circles, W_circles = figure_concentric_circles()
    fig1.savefig("notebooks/njw_fig1_circles.png", dpi=150, bbox_inches="tight")
    print("  → notebooks/njw_fig1_circles.png")

    # Figure 2
    print("Figure 2: Half-moons...")
    fig2 = figure_half_moons()
    fig2.savefig("notebooks/njw_fig2_moons.png", dpi=150, bbox_inches="tight")
    print("  → notebooks/njw_fig2_moons.png")

    # Figure 3
    print("Figure 3: Eigenvector embedding...")
    fig3 = figure_eigenvector_embedding(X_circles, W_circles)
    fig3.savefig("notebooks/njw_fig3_embedding.png", dpi=150, bbox_inches="tight")
    print("  → notebooks/njw_fig3_embedding.png")

    # Figure 4
    print("Figure 4: Eigenvalue spectrum & eigengap...")
    fig4 = figure_eigenvalue_spectrum()
    fig4.savefig("notebooks/njw_fig4_eigengap.png", dpi=150, bbox_inches="tight")
    print("  → notebooks/njw_fig4_eigengap.png")

    # Figure 5
    print("Figure 5: CASA — auditory stream segregation...")
    fig5 = figure_casa_stream_segregation()
    fig5.savefig("notebooks/njw_fig5_casa.png", dpi=150, bbox_inches="tight")
    print("  → notebooks/njw_fig5_casa.png")

    plt.close("all")
    print("\nDone. All figures saved to notebooks/")


if __name__ == "__main__":
    main()