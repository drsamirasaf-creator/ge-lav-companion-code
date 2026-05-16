"""
Stress scenario testing for GE-LAV portfolios.

Defines a small library of standard stress scenarios (mild downturn, recession,
GFC-equivalent, COVID-like rapid, persistent low, reverse stress) and a runner
that applies each scenario to a portfolio of funds, tracking mark-to-market,
exit decisions under L*(t), and P&L paths.

Used in:
  - Session 18 (platform architecture)
  - Capstone (project workshop)

Example
-------
>>> from gelav.stress import StressScenario, run_stress
>>> from gelav.term_structure import pi_of_L_T
>>>
>>> portfolio = [
...     {"name": "Fund A", "NAV": 100, "T_remaining": 5.0},
...     {"name": "Fund B", "NAV": 50,  "T_remaining": 3.0},
... ]
>>> scenario = StressScenario.gfc_equivalent()
>>> result = run_stress(portfolio, scenario)
>>> print(result["mtm_path"].shape)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence, Optional
import numpy as np

from .term_structure import pi_of_L_T
from .ou import simulate_ou


@dataclass
class StressScenario:
    """A named stress scenario on the L_t state.

    A scenario is described by an L-shock path: a sequence of L values at
    monthly grid points. The path can be deterministic (typical for "what-if"
    analysis) or simulated stochastically (typical for distribution analysis).

    Attributes
    ----------
    name : str
        Human-readable scenario name.
    L_path : np.ndarray
        Sequence of L values at each monthly grid point (length n_months+1).
    duration_months : int
        Total scenario length in months.
    description : str
        Short description of the economic interpretation.
    """
    name: str
    L_path: np.ndarray
    duration_months: int
    description: str = ""

    @classmethod
    def mild_downturn(cls, L_start: float = 0.0) -> "StressScenario":
        """L: 0 → -0.5 over 12 months. Tactical rebalance scenario."""
        n = 12
        path = np.linspace(L_start, -0.5, n + 1)
        return cls("Mild downturn", path, n,
                   "L drifts from 0 to -0.5 over 12 months. Used for tactical rebalance.")

    @classmethod
    def recession(cls, L_start: float = 0.0) -> "StressScenario":
        """L: 0 → -1.0 over 24 months. Liquidity planning scenario."""
        n = 24
        path = np.linspace(L_start, -1.0, n + 1)
        return cls("Recession", path, n,
                   "L drops from 0 to -1.0 over 24 months. Liquidity planning anchor.")

    @classmethod
    def gfc_equivalent(cls, L_start: float = 0.0) -> "StressScenario":
        """L: 0 → -1.5 over 18 months. GFC-equivalent tail scenario."""
        n = 18
        # Sharper initial drop, slower bottom
        t = np.arange(n + 1) / n
        path = L_start + (-1.5 - L_start) * (1 - np.exp(-3 * t)) / (1 - np.exp(-3))
        return cls("GFC-equivalent", path, n,
                   "L crashes to -1.5 over 18 months. Severe tail risk scenario.")

    @classmethod
    def covid_rapid(cls, L_start: float = 0.0) -> "StressScenario":
        """L: 0 → -1.2 in 3 months, recovers to -0.3 over next 3 months. Whipsaw."""
        n = 6
        path = np.array([L_start, -0.4, -1.0, -1.2, -0.9, -0.5, -0.3])
        return cls("COVID-like rapid", path, n,
                   "L crashes to -1.2 in 3 months, partial recovery over next 3.")

    @classmethod
    def persistent_low(cls, L_start: float = 0.0) -> "StressScenario":
        """L drifts to -0.5 and stays there for 60 months. Long stagnation."""
        n = 60
        # First 6 months: drift down. Next 54: stay at -0.5.
        path = np.concatenate([np.linspace(L_start, -0.5, 7), np.full(n - 6, -0.5)])
        return cls("Persistent low",  path, n,
                   "L drifts to -0.5 over 6 months, stays for 60 months total.")

    @classmethod
    def reverse_stress(cls, L_start: float = 0.0) -> "StressScenario":
        """L: 0 → +1.0 over 12 months. Capital release / rebalance opportunity."""
        n = 12
        path = np.linspace(L_start, 1.0, n + 1)
        return cls("Reverse stress", path, n,
                   "L rises to +1.0. Capital release / rebalance opportunity.")

    @classmethod
    def custom(cls, name: str, L_values: Sequence[float],
               description: str = "") -> "StressScenario":
        """Build a custom scenario from an explicit L-value sequence (monthly)."""
        path = np.asarray(L_values, dtype=float)
        return cls(name, path, len(path) - 1, description or f"Custom: {name}")

    @classmethod
    def stochastic_ou(cls, name: str = "Stochastic OU",
                      kappa: float = 0.45, sigma: float = 0.32, L_bar: float = 1.0,
                      L_start: float = 0.0, duration_months: int = 24,
                      seed: Optional[int] = None) -> "StressScenario":
        """Generate a single stochastic OU path (for Monte Carlo applications)."""
        T = duration_months / 12.0
        dt = 1.0 / 12.0  # monthly steps
        path = simulate_ou(L0=L_start, kappa=kappa, sigma=sigma, L_bar=L_bar,
                           T=T, dt=dt, n_paths=1, seed=seed)[0]
        return cls(name, path, duration_months, f"Stochastic OU realization (seed={seed}).")


def standard_scenarios(L_start: float = 0.0) -> dict[str, StressScenario]:
    """Return a dict of the 6 standard stress scenarios at a given starting L."""
    return {
        "mild": StressScenario.mild_downturn(L_start),
        "recession": StressScenario.recession(L_start),
        "gfc": StressScenario.gfc_equivalent(L_start),
        "covid": StressScenario.covid_rapid(L_start),
        "stagnation": StressScenario.persistent_low(L_start),
        "reverse": StressScenario.reverse_stress(L_start),
    }


def run_stress(portfolio: Sequence[dict],
               scenario: StressScenario,
               exit_boundary_fn=None,
               alpha: float = 0.045,
               beta: float = -0.025,
               gamma: float = 0.021) -> dict:
    """Apply a stress scenario to a portfolio, tracking MTM and exits.

    Parameters
    ----------
    portfolio : list of dicts
        Each fund: {"name": str, "NAV": float, "T_remaining": float (years)}.
    scenario : StressScenario
        The stress path to apply.
    exit_boundary_fn : callable, optional
        Function L*(t) → optimal exit threshold. If provided, funds whose
        current L falls below their L*(t) are marked as exiting.
    alpha, beta, gamma : floats
        π(L, T) coefficients. Defaults to course-calibrated values.

    Returns
    -------
    result : dict
        Keys: "mtm_path" (n_funds × n_months+1 array), "total_mtm" (n_months+1
        array), "exit_decisions" (n_funds × n_months+1 boolean array),
        "scenario", "portfolio".
    """
    L_path = scenario.L_path
    n_steps = len(L_path)
    n_funds = len(portfolio)

    mtm_path = np.zeros((n_funds, n_steps))
    exit_decisions = np.zeros((n_funds, n_steps), dtype=bool)

    for j, fund in enumerate(portfolio):
        NAV = fund["NAV"]
        T_rem_start = fund["T_remaining"]
        for i, L in enumerate(L_path):
            # Time remaining decreases linearly with the scenario clock
            t_elapsed_yr = i / 12.0
            T_rem = max(0.1, T_rem_start - t_elapsed_yr)
            pi = pi_of_L_T(L=L, T=T_rem, alpha=alpha, beta=beta, gamma=gamma)
            mtm_path[j, i] = NAV * (1.0 - pi)

            if exit_boundary_fn is not None:
                L_star = exit_boundary_fn(T_rem)
                if L < L_star:
                    exit_decisions[j, i] = True

    total_mtm = mtm_path.sum(axis=0)

    return {
        "mtm_path": mtm_path,
        "total_mtm": total_mtm,
        "exit_decisions": exit_decisions,
        "scenario": scenario,
        "portfolio": portfolio,
        "L_path": L_path,
        "initial_NAV": sum(f["NAV"] for f in portfolio),
    }


def summarize_stress(result: dict) -> dict:
    """Compute summary statistics for a stress run."""
    initial = result["initial_NAV"]
    final = result["total_mtm"][-1]
    drawdown = (result["total_mtm"].min() - initial) / initial
    recovery = (final - result["total_mtm"].min()) / initial
    n_exits = int(result["exit_decisions"].any(axis=1).sum())
    return {
        "scenario_name": result["scenario"].name,
        "initial_value": float(initial),
        "final_value": float(final),
        "max_drawdown_pct": float(drawdown * 100),
        "recovery_pct": float(recovery * 100),
        "n_funds_exiting": n_exits,
        "duration_months": result["scenario"].duration_months,
    }
