"""
Expectation Propagation via Kalman smoothing.

Implements Algorithm 1 from Wilkinson et al. (2019): EP inference in a
state-space Gaussian process model using a forward Kalman filter and a
backward Rauch-Tung-Striebel (RTS) smoother.

The state-space formulation converts GP inference from O(N³) to O(N) in
the number of time steps, making it practical for long audio signals.
The EP outer loop handles non-Gaussian likelihoods (e.g., Poisson
observations, censored data) by iteratively refining Gaussian
approximations to the likelihood at each site.

For Gaussian likelihoods the EP loop converges in a single pass — the
algorithm reduces to the standard Kalman smoother.

References
----------
Wilkinson et al. (2019). "End-to-end probabilistic inference for
  nonstationary audio analysis." arXiv:1901.11436.
Rasmussen & Williams (2006). Gaussian Processes for Machine Learning,
  §6.2 (state-space models).
"""

import numpy as np


def make_ou_ssm(dt, lengthscale=0.3, variance=1.0, noise_var=0.1):
    """
    Discretised state-space model for a Matérn-1/2 (Ornstein-Uhlenbeck) GP.

    Continuous form:  dx = −x/ℓ dt + √(2σ²/ℓ) dW
    Discrete form:    x_k = A x_{k−1} + q_k,   q_k ~ N(0, Q)

    Parameters
    ----------
    dt : float
        Time step between observations.
    lengthscale : float
        GP lengthscale ℓ.
    variance : float
        GP marginal variance σ².
    noise_var : float
        Observation noise variance.

    Returns
    -------
    A : float
        State transition coefficient.
    Q : float
        Process noise variance.
    H : ndarray, shape (1,)
        Observation matrix.
    P0 : float
        Initial state variance.
    noise_var : float
        Observation noise variance (passed through for convenience).
    """
    lam = 1.0 / lengthscale
    A = np.exp(-lam * dt)
    Q = variance * (1.0 - A ** 2)
    H = np.array([1.0])
    P0 = variance
    return A, Q, H, P0, noise_var


def ep_kalman_smoother(y, A, Q, H, P0, noise_var, n_iter=5):
    """
    Algorithm 1: EP inference via Kalman filtering and RTS smoothing.

    For a Gaussian likelihood p(y_k | f_k) = N(y_k; H f_k, σ²_n), EP is
    exact and converges in one pass.  The full EP loop is retained so the
    function generalises to non-Gaussian likelihoods (swap the site update
    rule).

    Parameters
    ----------
    y : ndarray, shape (T,)
        Observations.
    A : float
        State transition coefficient.
    Q : float
        Process noise variance.
    H : ndarray, shape (1,)
        Observation matrix (scalar measurement).
    P0 : float
        Initial state variance.
    noise_var : float
        Observation noise variance.
    n_iter : int
        Number of EP iterations.

    Returns
    -------
    history : list of dict
        One entry per EP iteration, each containing:

        - ``m_fwd``   (T,) — forward filter means
        - ``P_fwd``   (T,) — forward filter variances
        - ``m_smo``   (T,) — smoother means
        - ``P_smo``   (T,) — smoother variances
        - ``tau_cav`` (T,) — cavity precisions
        - ``nu_cav``  (T,) — cavity natural means
    """
    T = len(y)
    h = H[0]

    # EP site parameters: precision τ_k and precision-weighted mean ν_k
    tau = np.full(T, 1.0 / noise_var)
    nu = y / noise_var

    history = []

    for _ in range(n_iter):
        # --- Forward pass (Kalman filter) --------------------------------
        m_fwd = np.zeros(T)
        P_fwd = np.zeros(T)

        for k in range(T):
            # Predict
            if k == 0:
                m_pred, P_pred = 0.0, P0
            else:
                m_pred = A * m_fwd[k - 1]
                P_pred = A ** 2 * P_fwd[k - 1] + Q

            # Kalman gain and update (using EP site as pseudo-likelihood)
            S = h ** 2 * P_pred + 1.0 / tau[k]
            K = P_pred * h / S
            m_fwd[k] = m_pred + K * (y[k] - h * m_pred)
            P_fwd[k] = (1.0 - K * h) * P_pred

        # --- Backward pass (RTS smoother) --------------------------------
        m_smo = m_fwd.copy()
        P_smo = P_fwd.copy()

        for k in range(T - 2, -1, -1):
            P_pred_next = A ** 2 * P_fwd[k] + Q
            G = P_fwd[k] * A / P_pred_next
            m_smo[k] = m_fwd[k] + G * (m_smo[k + 1] - A * m_fwd[k])
            P_smo[k] = P_fwd[k] + G ** 2 * (P_smo[k + 1] - P_pred_next)

        # --- EP site update (Gaussian likelihood → exact) ----------------
        tau[:] = 1.0 / noise_var
        nu[:] = y / noise_var

        # --- Cavity parameters (for diagnostics) -------------------------
        tau_cav = np.zeros(T)
        nu_cav = np.zeros(T)
        for k in range(T):
            sigma2_smo = h ** 2 * P_smo[k]
            tau_cav[k] = 1.0 / sigma2_smo - tau[k]
            nu_cav[k] = m_smo[k] * h / sigma2_smo - nu[k]

        history.append(
            {
                "m_fwd": m_fwd.copy(),
                "P_fwd": P_fwd.copy(),
                "m_smo": m_smo.copy(),
                "P_smo": P_smo.copy(),
                "tau_cav": tau_cav.copy(),
                "nu_cav": nu_cav.copy(),
            }
        )

    return history
