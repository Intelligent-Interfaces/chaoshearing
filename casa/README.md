# casa — Computational Auditory Scene Analysis

Adaptive signal processing primitives for analysing, decomposing, and reconstructing complex acoustic scenes.

## Overview

The `casa` package provides five composable modules that together form a pipeline for probabilistic auditory signal processing:

```
signal → oscillators (transduction)
       → kernels (spectral decomposition)
       → kalman (temporal tracking)
       → graphs (scene structure)
       → diffusion (generative reconstruction)
```

Each module is self-contained and useful on its own. Together they implement the core computational loop of auditory scene analysis: **decompose** a complex mixture into structured components, **track** those components over time, **characterise** their relationships, and **reconstruct** or **generate** plausible scene configurations.

## Modules

### `kernels` — Spectral Mixture GPs

Probabilistic time-frequency analysis using Gaussian processes with spectral mixture kernels. Replaces fixed-resolution spectrograms with uncertainty-aware spectral estimates.

```python
from casa.kernels import gp_regression, spectral_density

mu, var = gp_regression(t_obs, y_obs, t_pred,
                        weights=[0.3, 0.5],
                        means=[400.0, 440.0],       # Hz
                        variances=[8000.0, 200.0])   # bandwidth²
```

Key functions:

- `spectral_mixture_kernel(tau, weights, means, variances)` — evaluate SM kernel at time lags
- `gp_regression(...)` — full GP posterior (mean + variance) at prediction points
- `spectral_density(weights, means, variances, f)` — implied power spectral density

### `kalman` — EP via Kalman Smoothing

State-space GP inference that scales linearly in time (O(N) vs O(N³) for direct GP). Forward Kalman filter + backward RTS smoother, wrapped in an Expectation Propagation loop for non-Gaussian likelihoods.

```python
from casa.kalman import ep_kalman_smoother, make_ou_ssm

A, Q, H, P0, nv = make_ou_ssm(dt=0.01, lengthscale=0.25, variance=1.0, noise_var=0.15)
history = ep_kalman_smoother(y, A, Q, H, P0, nv, n_iter=5)

smoother_mean = history[-1]["m_smo"]
smoother_var  = history[-1]["P_smo"]
```

### `graphs` — Similarity Graphs & Spectral Geometry

Build weighted graphs from any pairwise similarity function, then analyse their spectral properties (Laplacian eigenvalues, spectral gap, effective dimension). Works with any vocabulary: phonemes, timbral tokens, environmental sound classes, spectral templates.

```python
from casa.graphs import build_similarity_graph, graph_laplacian, spectral_analysis, spectral_gap

W = build_similarity_graph(items, similarity_fn, threshold=0.1)
L, degrees = graph_laplacian(W)
eigenvalues, eigenvectors = spectral_analysis(L, k=20)
gap = spectral_gap(eigenvalues)
```

### `diffusion` — Absorbing-State Discrete Diffusion

Structured sequence generation via discrete denoising diffusion. The forward process progressively masks tokens; the reverse process recovers them. Useful for generative reconstruction of partially observed auditory sequences.

```python
from casa.diffusion import AbsorbingDiffusion

diff = AbsorbingDiffusion(vocab_size=500, seq_len=16, n_steps=100)
x_t = diff.forward(x0, t=50)                    # corrupt
schedule = diff.masking_schedule(x0_batch)       # analyse schedule
cde = diff.cumulative_decoding_error(error_rates) # compound error
```

### `oscillators` — Hopf Bifurcation Cochlear Model

Nonlinear oscillator models of cochlear transduction. A single Hopf oscillator or a full filterbank with logarithmically spaced characteristic frequencies.

```python
from casa.oscillators import cochlear_filterbank

responses, freqs = cochlear_filterbank(signal, sr=16000, n_channels=64,
                                       f_low=20, f_high=8000, mu=0.0)
```

## Dependencies

- `numpy >= 1.24`
- `scipy >= 1.10`
- `matplotlib >= 3.7` (for visualisation only)

## References

- Wilkinson et al. (2019). "End-to-end probabilistic inference for nonstationary audio analysis." [arXiv:1901.11436](https://arxiv.org/abs/1901.11436)
- Bregman, A.S. (1990). _Auditory Scene Analysis_. MIT Press.
- Austin et al. (2021). "Structured denoising diffusion models in discrete state-spaces." NeurIPS.
- Eguíluz et al. (2000). "Essential nonlinearities in hearing." Physical Review Letters.
- Chung, F.R.K. (1997). _Spectral Graph Theory_. AMS.
