"""GE-LAV Companion Code

Reusable mathematical primitives for the GE-LAV graduate finance course.

Modules
-------
ou               : Ornstein-Uhlenbeck process (simulate, calibrate)
term_structure   : pi(L, T) illiquidity premium term structure
metrics          : IRR, PME, LA-IRR, LA-PME
hjb              : Hamilton-Jacobi-Bellman solver, exit boundary L*(t)
fokker_planck    : Forward Kolmogorov equation solver
mfg              : Mean-field game fixed-point iteration
jensen           : Jensen bias formula
pigou            : Pigouvian tax tau*(L)
stress           : Stress scenario testing for portfolios
data             : Synthetic dataset generators

Examples
--------
>>> from gelav.ou import simulate_ou, calibrate_ou
>>> paths = simulate_ou(L0=0, kappa=0.45, sigma=0.32, L_bar=1.0,
...                     T=10.0, dt=1/252, n_paths=100)

Calibration parameters used throughout the book:
    kappa = 0.45  (mean reversion speed, half-life ~1.54 yrs)
    sigma = 0.32  (volatility)
    L_bar = 1.0   (long-run mean of liquidity state)
"""

__version__ = "0.1.0"
__author__ = "Samir Asaf"

# Convenience re-exports
from gelav.ou import simulate_ou, calibrate_ou, stationary_distribution, half_life
from gelav.term_structure import pi_of_L, pi_of_L_T, fit_pi
from gelav.metrics import irr, pme, la_irr, la_pme, tvpi, dpi
from gelav.hjb import solve_hjb, exit_boundary
from gelav.fokker_planck import solve_fp, stationary_density
from gelav.jensen import jensen_bias, jensen_bias_annualized
from gelav.pigou import tau_star, welfare_gap_annual
from gelav.mfg import mfg_solver
from gelav.stress import StressScenario, standard_scenarios, run_stress, summarize_stress

__all__ = [
    "simulate_ou", "calibrate_ou", "stationary_distribution", "half_life",
    "pi_of_L", "pi_of_L_T", "fit_pi",
    "irr", "pme", "la_irr", "la_pme", "tvpi", "dpi",
    "solve_hjb", "exit_boundary",
    "solve_fp", "stationary_density",
    "jensen_bias", "jensen_bias_annualized",
    "tau_star", "welfare_gap_annual",
    "mfg_solver",
    "StressScenario", "standard_scenarios", "run_stress", "summarize_stress",
]
