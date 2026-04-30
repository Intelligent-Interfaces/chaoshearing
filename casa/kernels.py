"""
Spectral mixture kernels and Gaussian process inference.

The spectral mixture (SM) kernel is a flexible stationary covariance function
whose spectral density is a mixture of Gaussians. By the Bochner / Wiener-
Khinchin theorem, any stationary covariance can be approximated this way,
making SM kernels a universal building block for probabilistic time-frequency
analysis.

Instead of the traditional pipeline (STFT → spectrogram → features), GP
inference with SM kernels jointly estimates time-frequency structure and
quantifies uncertainty — closer to how the cochlea processes sound than a
fixed-resolution Fourier transform.

References
----------
Wilson & Adams (2013). "Gaussian process kernels for pattern discovery
  and extrapolation." ICML.
Wilkinson et al. (2019). "End-to-end probabilistic inference for
  nonstationary audio analysis." arXiv:1901.11436.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def spectral_mixture_kernel(tau, weights, means, variances):
    """
    Evaluate the spectral mixture kernel at given time lags.

    k(τ) = Σ_q  w_q · exp(−2π² σ²_q τ²) · cos(2π μ_q τ)

    Each component q contributes a cosine (carrier at frequency μ_q)
    modulated by a Gaussian envelope (bandwidth controlled by σ²_q).

    Parameters
    ----------
    tau : ndarray, shape (...)
        Time lags (seconds). Any shape; computation is element-wise.
    weights : array-like, shape (Q,)
        Mixture weights (amplitudes of each spectral component).
    means : array-like, shape (Q,)
        Spectral means (centre frequencies in Hz).
    variances : array-like, shape (Q,)
        Spectral variances (bandwidth² in Hz²).

    Returns
    -------
    K : ndarray, same shape as *tau*
        Kernel values.
    """
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)

    K = np.zeros_like(tau, dtype=float)
    for w, mu, sigma2 in zip(weights, means, variances):
        K += w * np.exp(-2.0 * np.pi ** 2 * sigma2 * tau ** 2) * np.cos(
            2.0 * np.pi * mu * tau
        )
    return K


def build_covariance_matrix(t, weights, means, variances, noise_var=1e-4):
    """
    Assemble the full covariance matrix K(t_i, t_j) + σ²_n I.

    Parameters
    ----------
    t : ndarray, shape (N,)
        Observation times.
    weights, means, variances : array-like, shape (Q,)
        SM kernel hyper-parameters.
    noise_var : float
        Observation noise variance σ²_n.

    Returns
    -------
    K : ndarray, shape (N, N)
        Positive-definite covariance matrix.
    """
    tau = t[:, None] - t[None, :]
    K = spectral_mixture_kernel(tau, weights, means, variances)
    K += noise_var * np.eye(len(t))
    return K


def gp_regression(t_obs, y_obs, t_pred, weights, means, variances,
                  noise_var=1e-4):
    """
    Standard GP regression with a spectral mixture kernel.

    Computes the posterior mean and marginal variance at prediction points
    given noisy observations.  Complexity is O(N³) in the number of
    observations, so this is suitable for short segments (< ~2000 points).
    For longer signals, use the state-space formulation in ``kalman.py``.

    Parameters
    ----------
    t_obs : ndarray, shape (N,)
        Observation times.
    y_obs : ndarray, shape (N,)
        Observed signal values.
    t_pred : ndarray, shape (M,)
        Prediction times.
    weights, means, variances : array-like, shape (Q,)
        SM kernel hyper-parameters.
    noise_var : float
        Observation noise variance.

    Returns
    -------
    mu : ndarray, shape (M,)
        Posterior mean at prediction points.
    var : ndarray, shape (M,)
        Posterior marginal variance at prediction points.
    """
    K_obs = build_covariance_matrix(t_obs, weights, means, variances, noise_var)

    tau_cross = t_pred[:, None] - t_obs[None, :]
    K_cross = spectral_mixture_kernel(tau_cross, weights, means, variances)

    tau_pred = t_pred[:, None] - t_pred[None, :]
    K_pred = spectral_mixture_kernel(tau_pred, weights, means, variances)

    L = cho_factor(K_obs)
    alpha = cho_solve(L, y_obs)
    mu = K_cross @ alpha

    v = cho_solve(L, K_cross.T)
    var = np.diag(K_pred) - np.sum(K_cross * v.T, axis=1)
    var = np.maximum(var, 0.0)

    return mu, var


def spectral_density(weights, means, variances, f):
    """
    Evaluate the power spectral density implied by SM kernel parameters.

    S(f) = Σ_q  w_q [ N(f; μ_q, σ²_q) + N(f; −μ_q, σ²_q) ]

    Parameters
    ----------
    weights, means, variances : array-like, shape (Q,)
        SM kernel hyper-parameters.
    f : ndarray, shape (...)
        Frequencies (Hz) at which to evaluate.

    Returns
    -------
    S : ndarray, same shape as *f*
        Spectral density values.
    """
    weights = np.asarray(weights, dtype=float)
    means = np.asarray(means, dtype=float)
    variances = np.asarray(variances, dtype=float)

    S = np.zeros_like(f, dtype=float)
    for w, mu, sigma2 in zip(weights, means, variances):
        sigma = np.sqrt(sigma2)
        S += w * (
            np.exp(-0.5 * ((f - mu) / sigma) ** 2)
            + np.exp(-0.5 * ((f + mu) / sigma) ** 2)
        )
    return S
