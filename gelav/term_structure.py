"""Illiquidity premium term structure pi(L, T).

The cross-sectional model used throughout the book is:

    pi(L, T) = (alpha + beta * L + gamma * L^2) * sqrt(T)

evaluated as a fraction of NAV. The square-root T scaling is empirical;
the (alpha, beta, gamma) parameters are calibrated from secondary market
discount data (see Lazard PCA quarterly reports).

Calibrated values (book Table 5.2):
    alpha =  0.045   (baseline premium at L=0)
    beta  = -0.025   (linear sensitivity)
    gamma =  0.021   (convexity)

Reference: Book Chapters 5 and 17.
"""
import numpy as np
from scipy.optimize import least_squares


def pi_of_L(L, alpha=0.045, beta=-0.025, gamma=0.021):
    """Single-horizon illiquidity premium pi(L).

    pi(L) = alpha + beta * L + gamma * L^2.

    Parameters
    ----------
    L : float or array_like
        Liquidity state(s).
    alpha, beta, gamma : float
        Coefficients. Defaults are the book's calibrated values.

    Returns
    -------
    Same shape as L. Fraction of NAV (e.g. 0.045 = 4.5 percent).
    """
    L = np.asarray(L, dtype=float)
    return alpha + beta * L + gamma * L ** 2


def pi_of_L_T(L, T, alpha=0.045, beta=-0.025, gamma=0.021,
              t_scaling="sqrt"):
    """Term-structure illiquidity premium pi(L, T).

    pi(L, T) = pi(L) * f(T)  where f(T) depends on `t_scaling`.

    Parameters
    ----------
    L : float or array_like
        Liquidity state(s).
    T : float or array_like
        Time to maturity in years. Must be broadcastable with L.
    alpha, beta, gamma : float
        Coefficients.
    t_scaling : str
        One of 'sqrt' (default, empirical), 'linear', 'none'.

    Returns
    -------
    Same shape as broadcast(L, T).
    """
    L = np.asarray(L, dtype=float)
    T = np.asarray(T, dtype=float)
    base = pi_of_L(L, alpha, beta, gamma)
    if t_scaling == "sqrt":
        f = np.sqrt(np.maximum(T, 0))
    elif t_scaling == "linear":
        f = T
    elif t_scaling == "none":
        f = np.ones_like(T)
    else:
        raise ValueError(f"unknown t_scaling: {t_scaling!r}")
    return base * f


def fit_pi(L, T, pi_observed, t_scaling="sqrt"):
    """Calibrate (alpha, beta, gamma) from observed secondary discounts.

    Solves the least-squares problem:

        min sum_i (pi_obs_i - (alpha + beta*L_i + gamma*L_i^2) * f(T_i))^2

    Parameters
    ----------
    L : array_like
        Observed L_t at each trade.
    T : array_like
        Observed time-to-maturity at each trade.
    pi_observed : array_like
        Observed discount as fraction of NAV.
    t_scaling : str
        Same as in `pi_of_L_T`.

    Returns
    -------
    dict with keys 'alpha', 'beta', 'gamma', 'r_squared', 'residuals'.
    """
    L = np.asarray(L, dtype=float)
    T = np.asarray(T, dtype=float)
    y = np.asarray(pi_observed, dtype=float)
    if not (len(L) == len(T) == len(y)):
        raise ValueError("L, T, pi_observed must have the same length")

    if t_scaling == "sqrt":
        f = np.sqrt(np.maximum(T, 0))
    elif t_scaling == "linear":
        f = T
    elif t_scaling == "none":
        f = np.ones_like(T)
    else:
        raise ValueError(f"unknown t_scaling: {t_scaling!r}")

    # Linear in (alpha, beta, gamma) for fixed t_scaling. Use OLS via lstsq.
    # Design matrix: [f, L*f, L^2*f]
    X = np.column_stack([f, L * f, L ** 2 * f])
    coefs, residuals_sq, rank, _ = np.linalg.lstsq(X, y, rcond=None)
    alpha, beta, gamma = coefs

    y_hat = X @ coefs
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha": float(alpha),
        "beta": float(beta),
        "gamma": float(gamma),
        "r_squared": float(r_squared),
        "residuals": y - y_hat,
        "n_obs": int(len(y)),
    }
