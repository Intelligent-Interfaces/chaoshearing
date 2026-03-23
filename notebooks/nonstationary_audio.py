"""
Nonstationary Audio Analysis with Spectral Mixture Gaussian Processes
=====================================================================

A demonstration of probabilistic time-frequency analysis for audio signals,
inspired by Wilkinson et al. (2019) "End-to-End Probabilistic Inference for
Nonstationary Audio Analysis" (arXiv:1901.11436).

Instead of the traditional pipeline (STFT → spectrogram → features), we model
audio signals directly with Gaussian processes using spectral mixture kernels.
This lets us:
  - Jointly estimate time-frequency structure and decompose sources
  - Handle nonstationarity naturally (the kernel parameters vary over time)
  - Quantify uncertainty in our spectral estimates

This script demonstrates the core ideas on synthetic signals before scaling
to real audio. It requires: numpy, scipy, matplotlib.

Usage:
    python nonstationary_audio.py

Erick Oduniyi — Chaos Hearing Project
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves PNG without opening a window
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
from scipy.linalg import cho_solve, cho_factor


# ---------------------------------------------------------------------------
# 1. Synthetic signal generation
# ---------------------------------------------------------------------------

def make_chirp(t, f0, f1, amplitude=1.0):
    """Linear chirp from f0 to f1 Hz over the duration of t."""
    phase = 2 * np.pi * (f0 * t + (f1 - f0) / (2 * t[-1]) * t**2)
    return amplitude * np.sin(phase)


def make_duet_signal(duration=2.0, sr=8000):
    """
    Create a synthetic 'duet' — two overlapping nonstationary sources.

    Source A: a chirp sweeping 200 → 600 Hz (like a rising voice)
    Source B: a pulsed tone at 440 Hz with amplitude modulation (like a bowed string)

    Returns t, mixture, source_a, source_b
    """
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # Source A: chirp
    source_a = make_chirp(t, 200, 600, amplitude=0.6)

    # Source B: amplitude-modulated tone (tremolo at 5 Hz)
    tremolo = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
    source_b = 0.5 * tremolo * np.sin(2 * np.pi * 440 * t)

    # Add a little noise (the cochlea operates in noise)
    noise = 0.05 * np.random.randn(len(t))

    mixture = source_a + source_b + noise
    return t, mixture, source_a, source_b


# ---------------------------------------------------------------------------
# 2. Spectral Mixture Kernel
# ---------------------------------------------------------------------------

def spectral_mixture_kernel(tau, weights, means, variances):
    """
    Spectral Mixture (SM) kernel — a flexible stationary kernel whose spectral
    density is a mixture of Gaussians. This means it can model any stationary
    covariance structure (by the Bochner/Wiener-Khinchin theorem).

    k(τ) = Σ_q w_q · exp(-2π²σ²_q τ²) · cos(2πμ_q τ)

    Parameters
    ----------
    tau : array, shape (n,) or (n, m)
        Time lags
    weights : array, shape (Q,)
        Mixture weights (amplitudes)
    means : array, shape (Q,)
        Spectral means (frequencies in Hz)
    variances : array, shape (Q,)
        Spectral variances (bandwidth² in Hz²)

    Returns
    -------
    K : array, same shape as tau
        Kernel values
    """
    K = np.zeros_like(tau, dtype=float)
    for w, mu, sigma2 in zip(weights, means, variances):
        K += w * np.exp(-2 * np.pi**2 * sigma2 * tau**2) * np.cos(2 * np.pi * mu * tau)
    return K


def build_covariance_matrix(t, weights, means, variances, noise_var=1e-4):
    """Build the full covariance matrix K(t_i, t_j) + σ²_n I."""
    n = len(t)
    tau = t[:, None] - t[None, :]  # (n, n) pairwise differences
    K = spectral_mixture_kernel(tau, weights, means, variances)
    K += noise_var * np.eye(n)
    return K


# ---------------------------------------------------------------------------
# 3. GP Regression with SM Kernel (small-scale demo)
# ---------------------------------------------------------------------------

def gp_spectral_analysis(t_obs, y_obs, t_pred, weights, means, variances,
                          noise_var=1e-4):
    """
    Standard GP regression using a spectral mixture kernel.

    For real audio (hundreds of thousands of points), you'd use the state-space
    formulation from Wilkinson et al. Here we do direct GP regression on a
    short segment to illustrate the idea.

    Returns posterior mean and variance at t_pred.
    """
    # Covariance matrices
    K_obs = build_covariance_matrix(t_obs, weights, means, variances, noise_var)
    tau_pred_obs = t_pred[:, None] - t_obs[None, :]
    K_pred_obs = spectral_mixture_kernel(tau_pred_obs, weights, means, variances)
    tau_pred = t_pred[:, None] - t_pred[None, :]
    K_pred = spectral_mixture_kernel(tau_pred, weights, means, variances)

    # Cholesky solve
    L = cho_factor(K_obs)
    alpha = cho_solve(L, y_obs)
    mu_pred = K_pred_obs @ alpha

    v = cho_solve(L, K_pred_obs.T)
    var_pred = np.diag(K_pred) - np.sum(K_pred_obs * v.T, axis=1)
    var_pred = np.maximum(var_pred, 0)  # numerical floor

    return mu_pred, var_pred


# ---------------------------------------------------------------------------
# 4. Spectral density visualization
# ---------------------------------------------------------------------------

def plot_spectral_density(weights, means, variances, f_max=1000, label=None):
    """Plot the spectral density implied by SM kernel parameters."""
    f = np.linspace(0, f_max, 2000)
    S = np.zeros_like(f)
    for w, mu, sigma2 in zip(weights, means, variances):
        sigma = np.sqrt(sigma2)
        S += w * (np.exp(-0.5 * ((f - mu) / sigma)**2) +
                  np.exp(-0.5 * ((f + mu) / sigma)**2))
    plt.plot(f, S, label=label)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Spectral Density")


# ---------------------------------------------------------------------------
# 5. Main demonstration
# ---------------------------------------------------------------------------

def main():
    print("Chaos Hearing — Nonstationary Audio Analysis Demo")
    print("=" * 52)

    # --- Generate synthetic duet ---
    duration = 2.0
    sr = 8000
    t, mixture, source_a, source_b = make_duet_signal(duration, sr)

    # --- Traditional STFT for comparison ---
    f_stft, t_stft, Zxx = stft(mixture, fs=sr, nperseg=256, noverlap=224)

    # --- SM kernel parameters (hand-tuned to match our synthetic signal) ---
    # Component 1: the chirp (broad, centered ~400 Hz)
    # Component 2: the 440 Hz tone (narrow)
    weights_full = np.array([0.3, 0.5])
    means_full = np.array([400.0, 440.0])
    variances_full = np.array([8000.0, 200.0])

    # --- GP regression on a short segment (direct GP is O(n³)) ---
    # Take a 50ms window for the GP demo
    window_dur = 0.05
    window_samples = int(sr * window_dur)
    start_idx = int(0.5 * sr)  # start at t=0.5s
    t_window = t[start_idx:start_idx + window_samples]
    y_window = mixture[start_idx:start_idx + window_samples]

    # Predict on a denser grid
    t_dense = np.linspace(t_window[0], t_window[-1], window_samples * 2)

    print(f"\nRunning GP regression on {window_dur*1000:.0f}ms window "
          f"({window_samples} samples)...")

    mu_pred, var_pred = gp_spectral_analysis(
        t_window, y_window, t_dense,
        weights_full, means_full, variances_full,
        noise_var=0.05**2
    )
    std_pred = np.sqrt(var_pred)

    print("Done.\n")

    # --- Plotting ---
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Chaos Hearing — Nonstationary Audio Analysis", fontsize=14)

    # (0,0) Full mixture waveform
    ax = axes[0, 0]
    ax.plot(t, mixture, linewidth=0.3, color="steelblue", alpha=0.8)
    ax.set_title("Mixture Signal (Synthetic Duet)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.axvspan(t_window[0], t_window[-1], alpha=0.15, color="orange",
               label="GP window")
    ax.legend(fontsize=8)

    # (0,1) Individual sources
    ax = axes[0, 1]
    ax.plot(t, source_a, linewidth=0.4, alpha=0.7, label="Chirp (200→600 Hz)")
    ax.plot(t, source_b, linewidth=0.4, alpha=0.7, label="Tone (440 Hz + tremolo)")
    ax.set_title("Ground Truth Sources")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)

    # (1,0) STFT spectrogram
    ax = axes[1, 0]
    ax.pcolormesh(t_stft, f_stft, np.abs(Zxx), shading="gouraud", cmap="magma")
    ax.set_title("STFT Spectrogram")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 1000)

    # (1,1) Spectral density of SM kernel
    ax = axes[1, 1]
    plt.sca(ax)
    plot_spectral_density(weights_full, means_full, variances_full,
                          f_max=1000, label="SM kernel PSD")
    ax.set_title("Spectral Mixture Kernel — Implied Spectral Density")
    ax.legend(fontsize=8)
    ax.axvline(440, color="gray", linestyle="--", alpha=0.5, label="440 Hz")
    ax.axvline(400, color="gray", linestyle=":", alpha=0.5, label="~chirp center")

    # (2,0) GP posterior on window
    ax = axes[2, 0]
    ax.plot(t_window * 1000, y_window, "k.", markersize=2, alpha=0.4,
            label="Observed")
    ax.plot(t_dense * 1000, mu_pred, "steelblue", linewidth=1.2,
            label="GP posterior mean")
    ax.fill_between(t_dense * 1000, mu_pred - 2*std_pred, mu_pred + 2*std_pred,
                    alpha=0.2, color="steelblue", label="±2σ")
    ax.set_title(f"GP Regression ({window_dur*1000:.0f}ms window at t=0.5s)")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Amplitude")
    ax.legend(fontsize=8)

    # (2,1) Residual / uncertainty
    ax = axes[2, 1]
    # Interpolate observed to dense grid for residual
    y_interp = np.interp(t_dense, t_window, y_window)
    residual = y_interp - mu_pred
    ax.plot(t_dense * 1000, residual, color="coral", linewidth=0.5, alpha=0.7)
    ax.fill_between(t_dense * 1000, -2*std_pred, 2*std_pred,
                    alpha=0.15, color="steelblue", label="±2σ bounds")
    ax.set_title("Residual & Uncertainty Bounds")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Residual")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("notebooks/nonstationary_audio_demo.png", dpi=150,
                bbox_inches="tight")

    print("Figure saved to notebooks/nonstationary_audio_demo.png")
    print("\nKey takeaway: The spectral mixture GP captures the frequency")
    print("content of the signal probabilistically — with uncertainty —")
    print("rather than through a fixed-resolution spectrogram. This is")
    print("closer to how the cochlea processes sound: a bank of tuned,")
    print("nonlinear oscillators that continuously adapt to the input.")


if __name__ == "__main__":
    main()
