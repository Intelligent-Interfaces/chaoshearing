"""
casa — Computational Auditory Scene Analysis
=============================================

Adaptive signal processing primitives for analyzing, decomposing, and
reconstructing complex acoustic scenes. Built on four pillars:

1. **Spectral mixture Gaussian processes** — probabilistic time-frequency
   analysis that replaces fixed-resolution spectrograms with uncertainty-
   aware spectral estimates (kernels, inference).

2. **State-space inference** — Expectation Propagation via Kalman smoothing
   for scalable, online temporal signal processing (kalman).

3. **Similarity graphs & spectral geometry** — graph Laplacian analysis of
   structured similarity (phonetic, spectral, semantic) for understanding
   the connectivity and clustering of token/event vocabularies (graphs).

4. **Discrete diffusion** — absorbing-state diffusion models for structured
   sequence generation and denoising under temporal constraints (diffusion).

5. **Nonlinear oscillators** — Hopf bifurcation models of cochlear
   transduction: the active, compressive, frequency-selective front end
   that converts pressure waves into neural representations (oscillators).

These components compose into full CASA pipelines:

    signal → oscillators (transduction)
           → kernels (spectral decomposition)
           → kalman (temporal tracking)
           → graphs (scene structure)
           → diffusion (generative reconstruction)

References
----------
- Wilkinson et al. (2019). "End-to-end probabilistic inference for
  nonstationary audio analysis." arXiv:1901.11436.
- Bregman, A.S. (1990). Auditory Scene Analysis. MIT Press.
- Austin et al. (2021). "Structured denoising diffusion models in
  discrete state-spaces." NeurIPS.
- Eguíluz et al. (2000). "Essential nonlinearities in hearing."
  Physical Review Letters.
"""

from .kernels import (
    spectral_mixture_kernel,
    build_covariance_matrix,
    gp_regression,
    spectral_density,
)
from .kalman import ep_kalman_smoother
from .graphs import (
    build_similarity_graph,
    graph_laplacian,
    spectral_analysis,
    spectral_gap,
    effective_dimension,
)
from .diffusion import AbsorbingDiffusion
from .oscillators import hopf_oscillator, cochlear_filterbank

__version__ = "0.1.0"
__all__ = [
    # kernels
    "spectral_mixture_kernel",
    "build_covariance_matrix",
    "gp_regression",
    "spectral_density",
    # kalman
    "ep_kalman_smoother",
    # graphs
    "build_similarity_graph",
    "graph_laplacian",
    "spectral_analysis",
    "spectral_gap",
    "effective_dimension",
    "spectral_clustering",
    # diffusion
    "AbsorbingDiffusion",
    # oscillators
    "hopf_oscillator",
    "cochlear_filterbank",
]
