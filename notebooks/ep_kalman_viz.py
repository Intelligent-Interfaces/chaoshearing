"""
EP via Kalman Smoothing — Algorithm 1 Visualization
=====================================================

Implements and visualizes Algorithm 1 from:
  Wilkinson et al. (2019) "End-to-End Probabilistic Inference for
  Nonstationary Audio Analysis" (arXiv:1901.11436)

The algorithm performs Expectation Propagation (EP) inference in a
state-space GP model using Kalman filtering (forward pass) and
RTS smoothing (backward pass).

We visualize:
  1. The synthetic signal and observation model
  2. Forward Kalman filter: mean & variance evolving in time
  3. Backward RTS smoother: refined mean & variance
  4. EP convergence: how cavity parameters (τ, ν) stabilise across iterations
  5. Final posterior vs ground truth

Usage:
    python notebooks/ep_kalman_viz.py

Erick Oduniyi — Chaos Hearing Project
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — saves PNG without opening a window
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch

np.random.seed(42)


# ---------------------------------------------------------------------------
# 1. State-space model (scalar, 1D latent state for clarity)
#    Discretised Matern-3/2 or simple AR(1) as a stand-in
# ---------------------------------------------------------------------------

def make_ssm(dt=0.01, lengthscale=0.3, variance=1.0, noise_var=0.1):
    """
    Discretised state-space model for a Matern-1/2 (OU) GP.

    Continuous: dx = -x/l dt + sqrt(2*var/l) dW
    Discrete:   x_k = A x_{k-1} + q_k,  q_k ~ N(0, Q)

    Returns A (scalar), Q (scalar), H (1,), P0 (scalar)
    """
    lam = 1.0 / lengthscale
    A = np.exp(-lam * dt)
    Q = variance * (1 - A**2)
    H = np.array([1.0])
    P0 = variance
    return A, Q, H, P0, noise_var


# ---------------------------------------------------------------------------
# 2. Synthetic signal
# ---------------------------------------------------------------------------

def make_signal(T=100, dt=0.01, freq=3.0, noise_var=0.1):
    """Smooth latent function + noisy observations."""
    t = np.arange(T) * dt
    f_true = np.sin(2 * np.pi * freq * t) * np.exp(-t * 0.5)
    y = f_true + np.sqrt(noise_var) * np.random.randn(T)
    return t, f_true, y


# ---------------------------------------------------------------------------
# 3. Algorithm 1: EP via Kalman Smoothing
# ---------------------------------------------------------------------------

def ep_kalman_smoother(y, A, Q, H, P0, noise_var, n_iter=5):
    """
    Algorithm 1 from Wilkinson et al. (2019).

    For a Gaussian likelihood p(y_k | f_k) = N(y_k; H f_k, noise_var),
    EP is exact (reduces to standard Kalman smoother). We implement the
    full EP loop anyway to show convergence behaviour.

    Parameters
    ----------
    y         : (T,) observations
    A, Q      : scalar state transition and noise variance
    H         : (1,) measurement vector
    P0        : scalar initial variance
    noise_var : observation noise variance
    n_iter    : number of EP iterations

    Returns
    -------
    history : list of dicts, one per iteration, each containing
              forward and backward means/variances and cavity params
    """
    T = len(y)
    h = H[0]  # scalar measurement

    # EP site parameters (τ_k, ν_k) — precision and precision-weighted mean
    tau = np.ones(T) / noise_var   # initialise to likelihood precision
    nu  = y / noise_var            # initialise to likelihood nat. param

    history = []

    for ep_iter in range(n_iter):
        # ---- FORWARD PASS (Kalman filter) --------------------------------
        m_fwd = np.zeros(T)
        P_fwd = np.zeros(T)

        for k in range(T):
            # Predict
            if k == 0:
                m_pred = 0.0
                P_pred = P0
            else:
                m_pred = A * m_fwd[k-1]
                P_pred = A**2 * P_fwd[k-1] + Q

            # Update with EP site (treat site as pseudo-likelihood)
            # Cavity: remove site k from the marginal
            sigma2_pred = h**2 * P_pred
            tau_cav = 1.0 / sigma2_pred - tau[k]
            nu_cav  = m_pred * h / sigma2_pred - nu[k]

            # For Gaussian likelihood, EP update is exact:
            # site precision = 1/noise_var, site nat mean = y/noise_var
            tau[k] = 1.0 / noise_var
            nu[k]  = y[k] / noise_var

            # Kalman gain and update
            S = h**2 * P_pred + 1.0 / tau[k]
            K = P_pred * h / S
            m_fwd[k] = m_pred + K * (y[k] - h * m_pred)
            P_fwd[k] = (1 - K * h) * P_pred

        # ---- BACKWARD PASS (RTS smoother) --------------------------------
        m_smo = m_fwd.copy()
        P_smo = P_fwd.copy()

        for k in range(T-2, -1, -1):
            P_pred_next = A**2 * P_fwd[k] + Q
            G = P_fwd[k] * A / P_pred_next          # smoother gain
            m_smo[k] = m_fwd[k] + G * (m_smo[k+1] - A * m_fwd[k])
            P_smo[k] = P_fwd[k] + G**2 * (P_smo[k+1] - P_pred_next)

        # ---- CAVITY PARAMETERS (for visualisation) -----------------------
        tau_cav_all = np.zeros(T)
        nu_cav_all  = np.zeros(T)
        for k in range(T):
            sigma2_smo = h**2 * P_smo[k]
            tau_cav_all[k] = 1.0 / sigma2_smo - tau[k]
            nu_cav_all[k]  = m_smo[k] * h / sigma2_smo - nu[k]

        history.append({
            "iter":    ep_iter,
            "m_fwd":   m_fwd.copy(),
            "P_fwd":   P_fwd.copy(),
            "m_smo":   m_smo.copy(),
            "P_smo":   P_smo.copy(),
            "tau":     tau.copy(),
            "nu":      nu.copy(),
            "tau_cav": tau_cav_all.copy(),
            "nu_cav":  nu_cav_all.copy(),
        })

    return history


# ---------------------------------------------------------------------------
# 4. Plotting
# ---------------------------------------------------------------------------

def plot_algorithm(t, f_true, y, history):
    n_iter = len(history)
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "Algorithm 1: EP via Kalman Smoothing (Wilkinson et al., 2019)",
        fontsize=13, y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ---- (0, :) Signal overview ------------------------------------------
    ax_sig = fig.add_subplot(gs[0, :])
    ax_sig.scatter(t, y, s=6, color="gray", alpha=0.5, label="Observations yₖ")
    ax_sig.plot(t, f_true, "k--", linewidth=1.2, label="True latent f(t)")

    # Overlay forward and smoothed posteriors from last iteration
    last = history[-1]
    std_fwd = np.sqrt(last["P_fwd"])
    std_smo = np.sqrt(last["P_smo"])
    ax_sig.plot(t, last["m_fwd"], color="steelblue", linewidth=1,
                alpha=0.6, label="Filter mean (forward)")
    ax_sig.fill_between(t, last["m_fwd"]-2*std_fwd, last["m_fwd"]+2*std_fwd,
                        alpha=0.12, color="steelblue")
    ax_sig.plot(t, last["m_smo"], color="darkorange", linewidth=1.5,
                label="Smoother mean (backward)")
    ax_sig.fill_between(t, last["m_smo"]-2*std_smo, last["m_smo"]+2*std_smo,
                        alpha=0.15, color="darkorange")
    ax_sig.set_title("Signal, Observations, and Final EP Posterior")
    ax_sig.set_xlabel("Time")
    ax_sig.set_ylabel("Amplitude")
    ax_sig.legend(fontsize=8, ncol=4)

    # ---- (1, 0) Forward filter variance over time ------------------------
    ax_fwd = fig.add_subplot(gs[1, 0])
    for i, h in enumerate(history):
        alpha = 0.3 + 0.7 * (i / max(n_iter-1, 1))
        ax_fwd.plot(t, h["P_fwd"], linewidth=0.8,
                    color="steelblue", alpha=alpha,
                    label=f"iter {i+1}" if i in [0, n_iter-1] else None)
    ax_fwd.set_title("Forward Filter Variance Pₖ")
    ax_fwd.set_xlabel("Time")
    ax_fwd.set_ylabel("Variance")
    ax_fwd.legend(fontsize=7)

    # ---- (1, 1) Smoother variance over time ------------------------------
    ax_smo = fig.add_subplot(gs[1, 1])
    for i, h in enumerate(history):
        alpha = 0.3 + 0.7 * (i / max(n_iter-1, 1))
        ax_smo.plot(t, h["P_smo"], linewidth=0.8,
                    color="darkorange", alpha=alpha,
                    label=f"iter {i+1}" if i in [0, n_iter-1] else None)
    ax_smo.set_title("Smoother Variance Pₖ (RTS)")
    ax_smo.set_xlabel("Time")
    ax_smo.set_ylabel("Variance")
    ax_smo.legend(fontsize=7)

    # ---- (1, 2) EP convergence: mean squared change in smoother mean -----
    ax_conv = fig.add_subplot(gs[1, 2])
    diffs = []
    for i in range(1, n_iter):
        diff = np.mean((history[i]["m_smo"] - history[i-1]["m_smo"])**2)
        diffs.append(diff)
    if any(d > 0 for d in diffs):
        ax_conv.semilogy(range(1, n_iter), diffs, "o-", color="purple",
                         linewidth=1.5, markersize=5)
    else:
        ax_conv.plot(range(1, n_iter), diffs, "o-", color="purple",
                     linewidth=1.5, markersize=5)
        ax_conv.text(0.5, 0.5, "Converged in 1 pass\n(Gaussian likelihood → EP exact)",
                     transform=ax_conv.transAxes, ha="center", va="center",
                     fontsize=7, color="gray")
    ax_conv.set_title("EP Convergence\n(MSE between smoother means)")
    ax_conv.set_xlabel("EP Iteration")
    ax_conv.set_ylabel("Mean Squared Change (log)")
    ax_conv.set_xticks(range(1, n_iter))

    # ---- (2, 0) Cavity precision τ⁻ over time ----------------------------
    ax_tau = fig.add_subplot(gs[2, 0])
    for i, h in enumerate(history):
        alpha = 0.3 + 0.7 * (i / max(n_iter-1, 1))
        ax_tau.plot(t, h["tau_cav"], linewidth=0.8,
                    color="teal", alpha=alpha,
                    label=f"iter {i+1}" if i in [0, n_iter-1] else None)
    ax_tau.set_title("Cavity Precision τ⁻ₖ")
    ax_tau.set_xlabel("Time")
    ax_tau.set_ylabel("Precision")
    ax_tau.legend(fontsize=7)

    # ---- (2, 1) Cavity location ν⁻ over time ----------------------------
    ax_nu = fig.add_subplot(gs[2, 1])
    for i, h in enumerate(history):
        alpha = 0.3 + 0.7 * (i / max(n_iter-1, 1))
        ax_nu.plot(t, h["nu_cav"], linewidth=0.8,
                   color="crimson", alpha=alpha,
                   label=f"iter {i+1}" if i in [0, n_iter-1] else None)
    ax_nu.set_title("Cavity Location ν⁻ₖ")
    ax_nu.set_xlabel("Time")
    ax_nu.set_ylabel("Nat. mean param")
    ax_nu.legend(fontsize=7)

    # ---- (2, 2) Algorithm flow diagram -----------------------------------
    ax_flow = fig.add_subplot(gs[2, 2])
    ax_flow.set_xlim(0, 10)
    ax_flow.set_ylim(0, 10)
    ax_flow.axis("off")
    ax_flow.set_title("Algorithm 1 Flow", fontsize=9)

    boxes = [
        (5, 9.0, "Initialise\nτ, ν", "white"),
        (5, 7.2, "→ Forward Pass\nKalman Filter", "#d0e8f7"),
        (5, 5.4, "← Backward Pass\nRTS Smoother", "#fde8cc"),
        (5, 3.6, "Update EP Sites\n(τₖ, νₖ)", "#e8f7d0"),
        (5, 1.8, "Converged?\nReturn posterior", "#f0e8f7"),
    ]
    for (x, y_b, label, color) in boxes:
        ax_flow.add_patch(plt.Rectangle((x-2.8, y_b-0.6), 5.6, 1.1,
                          facecolor=color, edgecolor="gray", linewidth=0.8,
                          zorder=2))
        ax_flow.text(x, y_b, label, ha="center", va="center",
                     fontsize=7, zorder=3)

    # Arrows between boxes
    for i in range(len(boxes)-1):
        x1, y1 = boxes[i][0], boxes[i][1] - 0.6
        x2, y2 = boxes[i+1][0], boxes[i+1][1] + 0.55
        ax_flow.annotate("", xy=(x2, y2), xytext=(x1, y1),
                         arrowprops=dict(arrowstyle="->", color="gray",
                                         lw=1.0))
    # Loop-back arrow
    ax_flow.annotate("", xy=(7.9, 7.75), xytext=(7.9, 1.8),
                     arrowprops=dict(arrowstyle="->", color="steelblue",
                                     lw=1.0,
                                     connectionstyle="arc3,rad=-0.3"))
    ax_flow.text(9.2, 4.8, "EP\nloop", fontsize=6.5, color="steelblue",
                 ha="center")

    plt.savefig("notebooks/ep_kalman_viz.png", dpi=150, bbox_inches="tight")
    print("Figure saved to notebooks/ep_kalman_viz.png")


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    print("EP via Kalman Smoothing — Algorithm 1 Visualization")
    print("=" * 52)

    T = 120
    dt = 0.01
    noise_var = 0.15

    A, Q, H, P0, nv = make_ssm(dt=dt, lengthscale=0.25,
                                variance=1.0, noise_var=noise_var)
    t, f_true, y = make_signal(T=T, dt=dt, freq=2.5, noise_var=noise_var)

    print(f"State-space model: A={A:.4f}, Q={Q:.4f}, P0={P0:.4f}")
    print(f"Running EP for 6 iterations on {T} time steps...\n")

    history = ep_kalman_smoother(y, A, Q, H, P0, noise_var, n_iter=6)

    last = history[-1]
    rmse_fwd = np.sqrt(np.mean((last["m_fwd"] - f_true)**2))
    rmse_smo = np.sqrt(np.mean((last["m_smo"] - f_true)**2))
    print(f"RMSE — Filter:   {rmse_fwd:.4f}")
    print(f"RMSE — Smoother: {rmse_smo:.4f}  (smoother uses future data)")

    plot_algorithm(t, f_true, y, history)


if __name__ == "__main__":
    main()
