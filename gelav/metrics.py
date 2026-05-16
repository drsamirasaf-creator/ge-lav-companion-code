"""Performance metrics: IRR, PME, LA-IRR, LA-PME.

Each metric fixes a different defect of IRR:

- IRR:     basic but biased (timing, reinvestment, multiple roots).
- PME:     fixes timing bias via public benchmark comparison (Kaplan-Schoar 2005).
- LA-IRR:  fixes state bias via pi(L,T) cash flow adjustment.
- LA-PME:  combines both fixes (most rigorous).

References
----------
- Kaplan, Schoar (2005), Private Equity Performance.
- Book Chapter 7 (Asaf, alternative metrics).
"""
import numpy as np
from scipy.optimize import brentq


# -- IRR --------------------------------------------------------------------

def irr(cashflows, times=None, guess=0.1):
    """Internal rate of return.

    Solves for r such that  sum_i CF_i * (1 + r)^(-t_i) = 0.

    Parameters
    ----------
    cashflows : array_like
        Cash flows (sign convention: contributions negative, distributions positive).
    times : array_like or None
        Times in years for each cash flow. Defaults to 0, 1, 2, ...
    guess : float
        Initial guess for the root-finder.

    Returns
    -------
    float, the annualized IRR.
    """
    cf = np.asarray(cashflows, dtype=float)
    if times is None:
        t = np.arange(len(cf), dtype=float)
    else:
        t = np.asarray(times, dtype=float)
    if len(cf) != len(t):
        raise ValueError("cashflows and times must have the same length")

    def npv(r):
        return float(np.sum(cf / (1 + r) ** t))

    # Bracket via expansion if needed
    lo, hi = -0.99, 10.0
    try:
        return brentq(npv, lo, hi, xtol=1e-8)
    except ValueError:
        # No sign change in [-0.99, 10]; fall back to undefined
        return float("nan")


def tvpi(contributions, distributions, current_nav=0.0):
    """Total Value to Paid-In multiple.

    TVPI = (distributions + current_nav) / contributions.

    Convention: contributions and distributions are positive magnitudes
    (not signed cash flows).
    """
    c = np.sum(np.asarray(contributions, dtype=float))
    d = np.sum(np.asarray(distributions, dtype=float))
    if c <= 0:
        return float("nan")
    return float((d + current_nav) / c)


def dpi(contributions, distributions):
    """Distributions to Paid-In multiple (realized portion of TVPI)."""
    c = np.sum(np.asarray(contributions, dtype=float))
    d = np.sum(np.asarray(distributions, dtype=float))
    if c <= 0:
        return float("nan")
    return float(d / c)


# -- PME (Kaplan-Schoar) ----------------------------------------------------

def pme(cashflows, times, benchmark_index):
    """Public Market Equivalent (Kaplan-Schoar).

    PME = sum(D_t / R_t) / sum(C_t / R_t)
    where R_t is the benchmark total-return index level at time t,
    C_t is contributions (positive magnitude), D_t is distributions.

    PME > 1 means the fund outperformed the benchmark, after adjusting
    for the timing of cash flows.

    Parameters
    ----------
    cashflows : array_like
        Signed cash flows: contributions negative, distributions positive.
    times : array_like
        Times in years.
    benchmark_index : array_like
        Benchmark total-return index level at each time. Same length as cashflows.

    Returns
    -------
    float.
    """
    cf = np.asarray(cashflows, dtype=float)
    R = np.asarray(benchmark_index, dtype=float)
    if len(cf) != len(R):
        raise ValueError("cashflows and benchmark_index must have the same length")
    if np.any(R <= 0):
        raise ValueError("benchmark_index must be strictly positive")

    contributions = np.where(cf < 0, -cf, 0.0)
    distributions = np.where(cf > 0, cf, 0.0)

    numerator = float(np.sum(distributions / R))
    denominator = float(np.sum(contributions / R))
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


# -- LA-IRR -----------------------------------------------------------------

def la_irr(cashflows, times, L_path, pi_func=None):
    """Liquidity-Adjusted IRR.

    Adjusts each cash flow by (1 - pi(L_t, T_remaining)) for distributions,
    and (1 + pi(L_t, T_remaining)) for capital calls (cost of being long
    in stress). Then computes the IRR of adjusted cash flows.

    Parameters
    ----------
    cashflows : array_like
        Signed cash flows: contributions negative, distributions positive.
    times : array_like
        Times in years.
    L_path : array_like
        Observed (or estimated) L_t at each time. Same length as cashflows.
    pi_func : callable or None
        A function (L, T_remaining) -> pi. If None, uses the book default.

    Returns
    -------
    float, the liquidity-adjusted IRR.
    """
    from gelav.term_structure import pi_of_L_T
    cf = np.asarray(cashflows, dtype=float)
    t = np.asarray(times, dtype=float)
    L = np.asarray(L_path, dtype=float)
    if not (len(cf) == len(t) == len(L)):
        raise ValueError("cashflows, times, L_path must have the same length")
    if pi_func is None:
        pi_func = pi_of_L_T

    T_total = float(t[-1])
    T_remaining = np.maximum(T_total - t, 0)
    pi_values = np.array([pi_func(L_i, T_r) for L_i, T_r in zip(L, T_remaining)])

    # Adjustment: distributions get (1 - pi), contributions get (1 + pi).
    # We treat the contribution sign convention by adjusting magnitudes.
    adjusted = np.where(cf > 0, cf * (1 - pi_values), cf * (1 + pi_values))
    return irr(adjusted, t)


# -- LA-PME -----------------------------------------------------------------

def la_pme(cashflows, times, benchmark_index, L_path, pi_func=None):
    """Liquidity-Adjusted PME.

    Like PME, but each cash flow is first adjusted by the pi(L_t, T_remaining)
    factor. Combines the timing adjustment of PME with the state adjustment
    of LA-IRR.

    Returns
    -------
    float, the liquidity-adjusted PME.
    """
    from gelav.term_structure import pi_of_L_T
    cf = np.asarray(cashflows, dtype=float)
    t = np.asarray(times, dtype=float)
    R = np.asarray(benchmark_index, dtype=float)
    L = np.asarray(L_path, dtype=float)
    if not (len(cf) == len(t) == len(L) == len(R)):
        raise ValueError("all inputs must have the same length")
    if pi_func is None:
        pi_func = pi_of_L_T

    T_total = float(t[-1])
    T_remaining = np.maximum(T_total - t, 0)
    pi_values = np.array([pi_func(L_i, T_r) for L_i, T_r in zip(L, T_remaining)])

    adjusted = np.where(cf > 0, cf * (1 - pi_values), cf * (1 + pi_values))
    contributions = np.where(adjusted < 0, -adjusted, 0.0)
    distributions = np.where(adjusted > 0, adjusted, 0.0)

    numerator = float(np.sum(distributions / R))
    denominator = float(np.sum(contributions / R))
    if denominator <= 0:
        return float("nan")
    return numerator / denominator
