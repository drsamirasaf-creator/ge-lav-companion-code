"""Ornstein-Uhlenbeck process for the liquidity state L_t.

The SDE governing L_t under GE-LAV is:

    dL = kappa * (L_bar - L) * dt + sigma * dW

with stationary distribution Normal(L_bar, sigma^2 / (2*kappa)) and
half-life ln(2)/kappa.

References
----------
- Book Chapter 4 (OU process intuition)
- Karatzas-Shreve, Brownian Motion and Stochastic Calculus, Ch. 5
"""
import numpy as np
from scipy.optimize import minimize


def simulate_ou(L0, kappa, sigma, L_bar, T, dt=1/252, n_paths=1, seed=None):
    """Simulate paths of an OU process via Euler-Maruyama.

    Parameters
    ----------
    L0 : float
        Initial state.
    kappa : float
        Mean reversion speed (year^-1). Book value: 0.45.
    sigma : float
        Diffusion coefficient. Book value: 0.32.
    L_bar : float
        Long-run mean. Book value: 1.0.
    T : float
        Time horizon in years.
    dt : float
        Time step in years (default 1/252 = daily).
    n_paths : int
        Number of independent paths to simulate.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (n_paths, n_steps + 1)
        Simulated paths, including the initial state.
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(T / dt))
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = L0
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        eps = rng.standard_normal(n_paths)
        paths[:, t + 1] = (
            paths[:, t]
            + kappa * (L_bar - paths[:, t]) * dt
            + sigma * sqrt_dt * eps
        )
    return paths


def calibrate_ou(L_series, dt):
    """Estimate (kappa, sigma, L_bar) from an observed L_t series via MLE.

    Uses the exact OU transition density (Gaussian) for likelihood,
    avoiding Euler discretization bias.

    Parameters
    ----------
    L_series : array_like, shape (n,)
        Observed L_t values at equally spaced times.
    dt : float
        Spacing in years between observations.

    Returns
    -------
    dict with keys 'kappa', 'sigma', 'L_bar', 'log_likelihood'.
    """
    L = np.asarray(L_series, dtype=float)
    if L.ndim != 1 or len(L) < 3:
        raise ValueError("L_series must be 1-D with at least 3 observations")

    def neg_log_likelihood(params):
        kappa, sigma, L_bar = params
        if kappa <= 0 or sigma <= 0:
            return 1e10
        # Exact OU transition: L_{t+dt} | L_t ~ Normal(mu, var)
        # mu = L_bar + (L_t - L_bar) * exp(-kappa*dt)
        # var = (sigma^2 / (2*kappa)) * (1 - exp(-2*kappa*dt))
        decay = np.exp(-kappa * dt)
        var = (sigma ** 2 / (2 * kappa)) * (1 - decay ** 2)
        if var <= 0:
            return 1e10
        mu = L_bar + (L[:-1] - L_bar) * decay
        residuals = L[1:] - mu
        ll = -0.5 * np.sum(
            np.log(2 * np.pi * var) + residuals ** 2 / var
        )
        return -ll

    # Initial guess from method of moments
    L_bar_init = float(np.mean(L))
    sigma_init = float(np.std(L) * np.sqrt(2 * 0.5))  # assume kappa~0.5
    kappa_init = 0.5

    result = minimize(
        neg_log_likelihood,
        x0=[kappa_init, sigma_init, L_bar_init],
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 5000},
    )
    kappa, sigma, L_bar = result.x
    return {
        "kappa": float(kappa),
        "sigma": float(sigma),
        "L_bar": float(L_bar),
        "log_likelihood": float(-result.fun),
        "converged": bool(result.success),
    }


def stationary_distribution(kappa, sigma, L_bar):
    """Return (mean, std) of the OU stationary distribution.

    Stationary: Normal(L_bar, sigma^2 / (2*kappa)).
    """
    return float(L_bar), float(sigma / np.sqrt(2 * kappa))


def half_life(kappa):
    """Time for an OU deviation from L_bar to halve (in years)."""
    return float(np.log(2) / kappa)


def autocorrelation(kappa, lag):
    """Theoretical OU autocorrelation at given lag (years).

    Corr(L_t, L_{t+lag}) = exp(-kappa * lag) for a stationary OU.
    """
    return float(np.exp(-kappa * lag))
