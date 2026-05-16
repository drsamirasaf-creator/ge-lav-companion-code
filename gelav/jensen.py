"""Jensen bias calculations.

For f(r) = exp(-r*T) (the discount factor), Jensen's inequality gives:

    E[exp(-r T)] - exp(-E[r] T) ~ 0.5 * T^2 * Var(r) * exp(-E[r] T)

This is the bias DCF makes by plugging the expected discount rate into the
exponent rather than taking the expectation of the exponential.

Annualized as a fraction of the discount factor:

    annualized_bias ~ 0.5 * T * Var(r)

Reference: Book Chapter 23.
"""
import numpy as np


def jensen_bias(T, var_r, r_bar=0.08):
    """Total Jensen bias over horizon T as a dollar fraction of NAV.

    Formula:
        bias = 0.5 * T^2 * var_r * exp(-r_bar * T)

    Parameters
    ----------
    T : float or array_like
        Horizon in years.
    var_r : float or array_like
        Variance of the discount rate. (sigma_r squared.)
    r_bar : float
        Expected discount rate.

    Returns
    -------
    Same shape as broadcast(T, var_r). Absolute bias in PV per dollar.
    """
    T = np.asarray(T, dtype=float)
    var_r = np.asarray(var_r, dtype=float)
    return 0.5 * T ** 2 * var_r * np.exp(-r_bar * T)


def jensen_bias_annualized(T, sigma_r):
    """Annualized Jensen bias (fraction per year).

    Formula:
        annualized_bias = 0.5 * T * sigma_r^2

    This is the leading-order annualized version, ignoring the e^(-r T)
    discounting (often a small correction at moderate r).

    Parameters
    ----------
    T : float or array_like
        Horizon in years.
    sigma_r : float or array_like
        Standard deviation of discount rate.

    Returns
    -------
    Same shape as broadcast(T, sigma_r).
    """
    T = np.asarray(T, dtype=float)
    sigma_r = np.asarray(sigma_r, dtype=float)
    return 0.5 * T * sigma_r ** 2


def jensen_bias_exact(T, mu_r, sigma_r):
    """Exact Jensen bias for r ~ Normal(mu_r, sigma_r^2).

    For normally distributed r, the exact expectation is:
        E[exp(-r T)] = exp(-mu_r * T + 0.5 * sigma_r^2 * T^2)

    So:
        exact_bias = exp(-mu_r * T + 0.5 sigma_r^2 T^2) - exp(-mu_r * T)
                   = exp(-mu_r * T) * (exp(0.5 sigma_r^2 T^2) - 1)

    For small sigma_r^2 T^2, exp(0.5 sigma_r^2 T^2) - 1 ~ 0.5 sigma_r^2 T^2,
    recovering the second-order approximation.
    """
    T = np.asarray(T, dtype=float)
    sigma_r = np.asarray(sigma_r, dtype=float)
    mu_r = np.asarray(mu_r, dtype=float)
    return np.exp(-mu_r * T) * (np.exp(0.5 * sigma_r ** 2 * T ** 2) - 1)


# Calibrated bias by asset class (book Table 23.1)
ASSET_CLASS_BIAS = {
    # asset_class : (T_years, sigma_r)
    "VC":              (10, 0.05),
    "Growth Equity":   (8,  0.04),
    "Buyout":          (7,  0.035),
    "Infrastructure":  (12, 0.025),
    "Real Estate":     (9,  0.030),
    "Private Credit":  (5,  0.020),
    "Public Equity":   (1,  0.05),
}


def asset_class_bias_table():
    """Pretty table of annualized Jensen bias by asset class."""
    rows = []
    for ac, (T, sigma_r) in ASSET_CLASS_BIAS.items():
        bias = jensen_bias_annualized(T, sigma_r)
        rows.append((ac, T, sigma_r, float(bias) * 100))
    return rows  # list of (asset_class, T, sigma_r, bias_pct)
