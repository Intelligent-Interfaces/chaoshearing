"""
Absorbing-state discrete diffusion for structured sequence generation.

Discrete denoising diffusion probabilistic models (D3PM) operate on
sequences of tokens from a finite vocabulary.  The *absorbing* variant
uses a distinguished [MASK] token: the forward process progressively
replaces tokens with [MASK], and the reverse (generative) process
recovers the original tokens from a fully masked state.

In the context of auditory scene analysis, discrete diffusion provides a
principled framework for:

- **Generative reconstruction**: synthesising plausible completions of
  partially observed auditory sequences (e.g., filling in masked
  segments of a spectrogram or event stream).
- **Denoising schedules**: the noise schedule controls *when* each
  position is resolved, analogous to how temporal attention is allocated
  during real-time auditory processing.
- **Structured transition matrices**: the rate matrix R_t can encode
  domain-specific similarity (spectral, phonetic, timbral) so that
  corruption respects the structure of the vocabulary.

This module provides the forward process, schedule analysis, and
(optionally, if PyTorch is available) a minimal denoising network for
training and generation.

References
----------
Austin et al. (2021). "Structured denoising diffusion models in
  discrete state-spaces." NeurIPS (D3PM).
Campbell et al. (2022). "A continuous time framework for discrete
  denoising models." NeurIPS (CTMC formulation).
"""

import numpy as np


class AbsorbingDiffusion:
    """
    Absorbing-state discrete diffusion over token sequences.

    Parameters
    ----------
    vocab_size : int
        Size of the token vocabulary (including special tokens).
    seq_len : int
        Fixed sequence length.
    n_steps : int
        Number of diffusion time steps.
    mask_idx : int
        Index of the [MASK] token in the vocabulary.
    pad_idx : int
        Index of the [PAD] token (never masked).
    """

    def __init__(self, vocab_size, seq_len, n_steps=100,
                 mask_idx=1, pad_idx=0):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_steps = n_steps
        self.mask_idx = mask_idx
        self.pad_idx = pad_idx

        # Linear noise schedule: masking probability increases over time
        self.betas = np.linspace(0.01, 0.5, n_steps)
        self.alpha_bars = np.cumprod(1.0 - self.betas)

    def forward(self, x0, t):
        """
        Apply forward corruption at time step *t*.

        Each non-pad token is independently replaced by [MASK] with
        probability 1 − ᾱ_t.

        Parameters
        ----------
        x0 : ndarray, shape (batch, seq_len) or (seq_len,)
            Original token indices.
        t : int
            Diffusion time step (0 = clean, n_steps−1 = nearly all masked).

        Returns
        -------
        x_t : ndarray, same shape as *x0*
            Corrupted sequence.
        """
        x0 = np.asarray(x0)
        mask_prob = 1.0 - self.alpha_bars[t]
        mask = np.random.random(x0.shape) < mask_prob
        x_t = x0.copy()
        x_t[mask] = self.mask_idx
        x_t[x0 == self.pad_idx] = self.pad_idx
        return x_t

    def masking_schedule(self, x0_batch, n_samples=200):
        """
        Compute the empirical per-position masking rate across time steps.

        Parameters
        ----------
        x0_batch : ndarray, shape (N, seq_len)
            A batch of clean sequences.
        n_samples : int
            Number of sequences to average over.

        Returns
        -------
        schedule : ndarray, shape (n_steps, seq_len)
            Fraction of non-pad tokens masked at each (time, position).
        """
        n_samples = min(n_samples, len(x0_batch))
        schedule = np.zeros((self.n_steps, self.seq_len))

        for t in range(self.n_steps):
            masked_counts = np.zeros(self.seq_len)
            total_counts = np.zeros(self.seq_len)

            for i in range(n_samples):
                x0 = x0_batch[i : i + 1]
                x_t = self.forward(x0, t)[0]

                for pos in range(self.seq_len):
                    if x0[0, pos] != self.pad_idx:
                        total_counts[pos] += 1
                        if x_t[pos] == self.mask_idx:
                            masked_counts[pos] += 1

            nonzero = total_counts > 0
            schedule[t, nonzero] = masked_counts[nonzero] / total_counts[nonzero]

        return schedule

    def cumulative_decoding_error(self, error_per_step):
        """
        Compute cumulative decoding error (CDE) across time steps.

        CDE formalises how errors compound during the reverse (generative)
        process: each incorrectly denoised token propagates through
        subsequent steps.

        Parameters
        ----------
        error_per_step : ndarray, shape (n_steps,)
            Per-step error rate (e.g., from a trained denoiser).

        Returns
        -------
        cde : ndarray, shape (n_steps,)
            Cumulative error at each step.
        """
        cde = np.zeros(self.n_steps)
        cde[0] = error_per_step[0]
        for t in range(1, self.n_steps):
            cde[t] = cde[t - 1] + error_per_step[t] * (1.0 - cde[t - 1])
        return cde
