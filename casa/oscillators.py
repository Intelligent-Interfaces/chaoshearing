"""
Nonlinear oscillator models of cochlear transduction.

The inner ear is not a passive frequency analyser.  Each hair cell
operates near a supercritical Hopf bifurcation — a critical point where
a stable equilibrium gives way to self-sustained oscillation.  This
gives the cochlea extraordinary properties:

- ~1000× amplification of weak signals near the characteristic frequency
- Compressive nonlinearity spanning ~120 dB of dynamic range
- Sharp frequency selectivity (each basilar membrane location is tuned)
- Spontaneous otoacoustic emissions (the ear produces sound)

This module provides:

1. A single Hopf oscillator (the building block).
2. A cochlear filterbank: an array of oscillators with logarithmically
   spaced characteristic frequencies, modelling the tonotopic map of
   the basilar membrane.

The oscillator output is the *transduced* representation of the input
signal — the first stage of computational auditory scene analysis.

References
----------
Eguíluz et al. (2000). "Essential nonlinearities in hearing."
  Physical Review Letters, 84(22), 5232.
Kern & Stoop (2003). "Essential role of couplings between hearing
  nonlinearities." Physical Review Letters, 91(12), 128101.
"""

import numpy as np


def hopf_oscillator(signal, dt, mu=0.0, omega0=2 * np.pi * 440,
                    a=1.0, noise_std=0.0):
    """
    Simulate a single Hopf oscillator driven by an external signal.

    The normal form of the supercritical Hopf bifurcation with additive
    forcing:

        dz/dt = (μ + iω₀) z − (a + i/3) |z|² z + F(t) + η(t)

    where z(t) ∈ ℂ is the oscillator state, F(t) is the driving signal,
    and η(t) is complex white Gaussian noise.

    Parameters
    ----------
    signal : ndarray, shape (T,)
        Real-valued driving signal (acoustic pressure).
    dt : float
        Integration time step (seconds).
    mu : float
        Bifurcation parameter.  μ < 0 → stable (damped), μ > 0 →
        oscillating (limit cycle), μ ≈ 0 → critical (maximum
        sensitivity and compression).
    omega0 : float
        Natural angular frequency (rad/s).  Default 2π·440 ≈ 2764 rad/s
        (concert A).
    a : float
        Nonlinear damping coefficient.
    noise_std : float
        Standard deviation of additive complex noise.

    Returns
    -------
    z : ndarray, shape (T,), dtype complex128
        Complex oscillator trajectory.
    amplitude : ndarray, shape (T,)
        Instantaneous amplitude |z(t)|.
    phase : ndarray, shape (T,)
        Instantaneous phase arg(z(t)).
    """
    T = len(signal)
    z = np.zeros(T, dtype=complex)

    # Clamp amplitude to prevent numerical blow-up in explicit Euler.
    # Physical cochlear oscillators saturate; this mirrors that behaviour.
    # Near criticality (μ ≈ 0) the gain is enormous, so we cap at a
    # generous but finite ceiling.
    max_amp = 100.0

    for n in range(1, T):
        z_n = z[n - 1]
        abs2 = np.abs(z_n) ** 2
        dzdt = (mu + 1j * omega0) * z_n - (a + 1j / 3.0) * abs2 * z_n + signal[n - 1]
        if noise_std > 0:
            dzdt += noise_std * (np.random.randn() + 1j * np.random.randn())
        z[n] = z_n + dt * dzdt
        # Saturate to prevent divergence (mirrors cochlear compression)
        if np.abs(z[n]) > max_amp:
            z[n] = z[n] * (max_amp / np.abs(z[n]))

    amplitude = np.abs(z)
    phase = np.angle(z)
    return z, amplitude, phase


def cochlear_filterbank(signal, sr, n_channels=64, f_low=20.0, f_high=8000.0,
                        mu=0.0, a=1.0, noise_std=0.0):
    """
    Simulate a bank of Hopf oscillators modelling the cochlear response.

    Characteristic frequencies are spaced logarithmically from *f_low*
    to *f_high*, approximating the tonotopic map of the basilar membrane.

    Parameters
    ----------
    signal : ndarray, shape (T,)
        Input audio signal.
    sr : float
        Sample rate (Hz).
    n_channels : int
        Number of frequency channels.
    f_low, f_high : float
        Frequency range (Hz).
    mu : float
        Bifurcation parameter (same for all channels; set near 0 for
        critical tuning).
    a : float
        Nonlinear damping coefficient.
    noise_std : float
        Per-channel noise level.

    Returns
    -------
    responses : ndarray, shape (n_channels, T)
        Amplitude envelope |z(t)| for each channel.
    frequencies : ndarray, shape (n_channels,)
        Characteristic frequency of each channel (Hz).
    """
    dt = 1.0 / sr
    frequencies = np.geomspace(f_low, f_high, n_channels)
    T = len(signal)
    responses = np.zeros((n_channels, T))

    for ch, f0 in enumerate(frequencies):
        omega0 = 2.0 * np.pi * f0
        _, amplitude, _ = hopf_oscillator(signal, dt, mu=mu, omega0=omega0,
                                          a=a, noise_std=noise_std)
        responses[ch] = amplitude

    return responses, frequencies
