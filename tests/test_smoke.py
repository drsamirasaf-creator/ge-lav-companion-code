"""Smoke tests for the gelav package.

Run with: pytest tests/
These tests verify that each module's API works end-to-end with the
course-calibrated parameters. They are intentionally lightweight — full
numerical validation is in the notebooks.
"""
import numpy as np
import pytest


# ---------- OU process ----------
def test_simulate_ou_shape_and_stationarity():
    from gelav.ou import simulate_ou, half_life, stationary_distribution
    paths = simulate_ou(L0=0.0, kappa=0.45, sigma=0.32, L_bar=1.0,
                        T=10.0, dt=0.01, n_paths=100, seed=42)
    assert paths.shape == (100, 1001)
    # Long-run mean approaches L_bar
    assert abs(paths[:, -1].mean() - 1.0) < 0.2
    # Half-life is correct
    assert abs(half_life(0.45) - np.log(2)/0.45) < 1e-9
    # Stationary std is sigma/sqrt(2*kappa)
    _, std = stationary_distribution(kappa=0.45, sigma=0.32, L_bar=1.0)
    assert abs(std - 0.32 / np.sqrt(2 * 0.45)) < 1e-9


def test_calibrate_ou_recovers_params():
    from gelav.ou import simulate_ou, calibrate_ou
    paths = simulate_ou(L0=1.0, kappa=0.5, sigma=0.3, L_bar=1.0,
                        T=50.0, dt=1/12, n_paths=1, seed=0)
    est = calibrate_ou(paths[0], dt=1/12)
    # MLE recovers within reasonable tolerance for a single long path
    assert abs(est["kappa"] - 0.5) < 0.2
    assert abs(est["sigma"] - 0.3) < 0.05


# ---------- Term structure ----------
def test_pi_of_L_T():
    from gelav.term_structure import pi_of_L, pi_of_L_T
    # At L=0, T=5: pi = alpha * sqrt(T) for default coefficients (or whatever the model does)
    p0 = pi_of_L(L=0.0)
    p_stress = pi_of_L(L=-1.5)
    # Stress price should be much higher than normal
    assert p_stress > p0
    # Term structure should increase with T
    p_5 = pi_of_L_T(L=0.0, T=5.0)
    p_10 = pi_of_L_T(L=0.0, T=10.0)
    assert p_10 > p_5


# ---------- Performance metrics ----------
def test_irr_pme_metrics():
    from gelav.metrics import irr, pme, la_irr, la_pme, tvpi, dpi
    # Simple fund: -100 at t=0, +200 at t=5
    cf = np.array([-100, 0, 0, 0, 0, 200])
    times = np.array([0, 1, 2, 3, 4, 5])
    bench = np.array([1.0, 1.05, 1.10, 1.15, 1.20, 1.25])  # 5%/yr
    L_path = np.zeros(6)  # state L=0 throughout

    r = irr(cf)
    assert abs(r - ((200 / 100) ** (1/5) - 1)) < 0.01  # ~14.87%

    pme_val = pme(cf, times, bench)
    assert pme_val > 1.0  # Beat the benchmark


# ---------- HJB / exit boundary ----------
def test_hjb_exit_boundary_monotone():
    from gelav.hjb import solve_hjb, exit_boundary
    res = solve_hjb(kappa=0.45, sigma=0.32, L_bar=1.0, r=0.08, T_fund=10.0,
                    n_L=60, n_t=60)
    t_rem, L_star = exit_boundary(res)
    # L*(t) should be a sensible array (finite, non-empty)
    assert L_star.shape == t_rem.shape
    assert np.isfinite(L_star).all()
    # At least some values should be < L_bar (exit boundary is below long-run mean)
    assert np.any(L_star < 1.0)


# ---------- Fokker-Planck ----------
def test_fokker_planck_stationary_density_integrates():
    from gelav.fokker_planck import stationary_density
    L_grid = np.linspace(-3, 5, 400)
    p = stationary_density(kappa=0.45, sigma=0.32, L_bar=1.0, L_grid=L_grid)
    integral = float(np.trapezoid(p, L_grid))
    assert abs(integral - 1.0) < 0.02  # probability sums to 1


# ---------- Jensen bias ----------
def test_jensen_bias_orders():
    from gelav.jensen import jensen_bias_annualized
    # VC > Buyout > Credit
    vc = jensen_bias_annualized(T=10, sigma_r=0.05)
    buyout = jensen_bias_annualized(T=7, sigma_r=0.035)
    credit = jensen_bias_annualized(T=5, sigma_r=0.02)
    assert vc > buyout > credit
    # Magnitude reasonable
    assert 0.005 < vc < 0.05


# ---------- Pigouvian tax ----------
def test_pigouvian_schedule_shape():
    from gelav.pigou import tau_star, welfare_gap_annual
    # No tax in boom
    assert tau_star(L=1.0) == 0
    # Positive tax in stress
    assert tau_star(L=-1.0) > 0
    assert tau_star(L=-1.5) > tau_star(L=-1.0)
    # Bounded
    assert tau_star(L=-3.0) <= 0.15
    # Welfare gap reasonable
    gap = welfare_gap_annual(AUM_global=13e12, gap_per_year=0.023)
    assert 200e9 < gap < 400e9


# ---------- MFG solver ----------
def test_mfg_solver_returns_valid_result():
    from gelav.mfg import mfg_solver
    r = mfg_solver(kappa=0.45, sigma=0.32, L_bar=1.0, r=0.08, T_fund=10.0,
                   n_L=50, n_t=50, max_iter=10, verbose=False)
    assert "V" in r
    assert r["V"].ndim == 2
    assert "mu" in r
    assert r["mu"].ndim == 1
    assert "L_star" in r
    assert "converged" in r
    # V should be 2D with reasonable shape
    assert r["V"].shape[0] > 10 and r["V"].shape[1] > 10


# ---------- Stress scenarios ----------
def test_stress_standard_scenarios_run():
    from gelav.stress import standard_scenarios, run_stress, summarize_stress
    portfolio = [
        {"name": "A", "NAV": 100, "T_remaining": 5.0},
        {"name": "B", "NAV": 50,  "T_remaining": 3.0},
    ]
    scenarios = standard_scenarios()
    assert len(scenarios) == 6
    for key, scn in scenarios.items():
        res = run_stress(portfolio, scn)
        summary = summarize_stress(res)
        assert summary["scenario_name"] == scn.name
        assert summary["initial_value"] == 150.0


def test_stress_gfc_drawdown():
    from gelav.stress import StressScenario, run_stress, summarize_stress
    portfolio = [{"name": "X", "NAV": 100, "T_remaining": 5.0}]
    gfc = StressScenario.gfc_equivalent()
    res = run_stress(portfolio, gfc)
    summary = summarize_stress(res)
    # GFC should produce a significant drawdown
    assert summary["max_drawdown_pct"] < -10.0


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
