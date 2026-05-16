"""Synthetic data generators (Lazard-like secondary pricing, fund cash flows, L history).

All data sets here are deterministic given the seed, so notebooks
reproduce identically across machines.
"""
import numpy as np
import pandas as pd


def generate_L_history(n_years=30, dt=0.25, seed=42,
                        kappa=0.45, sigma=0.32, L_bar=1.0):
    """Generate a synthetic L_t history (quarterly steps by default).

    Returns
    -------
    pd.DataFrame with columns ['date', 'L_t'].
    """
    rng = np.random.default_rng(seed)
    n_steps = int(round(n_years / dt))
    L = np.empty(n_steps + 1)
    L[0] = L_bar
    sqrt_dt = np.sqrt(dt)
    for t in range(n_steps):
        L[t + 1] = L[t] + kappa * (L_bar - L[t]) * dt + sigma * sqrt_dt * rng.standard_normal()

    dates = pd.date_range(start="1996-01-01", periods=n_steps + 1, freq="QS")
    return pd.DataFrame({"date": dates, "L_t": L})


def generate_synthetic_secondary_pricing(n_obs=500, seed=42,
                                          alpha=0.045, beta=-0.025, gamma=0.021):
    """Generate Lazard-like secondary market pricing observations.

    Each observation: (vintage, L_t, T_remaining, discount, fund_type).
    discount = pi(L, T) + Gaussian noise.

    Returns
    -------
    pd.DataFrame.
    """
    rng = np.random.default_rng(seed)

    # Sample L from stationary OU distribution (skewed via realistic regime mix)
    # Use mixture: 70% normal regime, 20% mild stress, 10% deep stress
    regime = rng.choice([0, 1, 2], size=n_obs, p=[0.70, 0.20, 0.10])
    L = np.where(regime == 0, rng.normal(0.5, 0.3, n_obs),
         np.where(regime == 1, rng.normal(-0.5, 0.3, n_obs),
                                rng.normal(-1.3, 0.4, n_obs)))

    # Time remaining: uniform across 0 to 12 years
    T_remaining = rng.uniform(0.5, 12.0, n_obs)

    # Trade year: 2010 to 2024
    trade_year = rng.choice(range(2010, 2025), size=n_obs)

    # Fund type
    fund_type = rng.choice(
        ["Buyout", "VC", "Growth", "Infra", "RE", "Credit"],
        size=n_obs,
        p=[0.40, 0.15, 0.15, 0.12, 0.10, 0.08],
    )

    # True discount = pi(L, T) with sqrt T scaling
    pi_true = (alpha + beta * L + gamma * L ** 2) * np.sqrt(T_remaining)
    # Add measurement noise (Lazard-style observation error ~1-2 pp)
    noise = rng.normal(0, 0.015, n_obs)
    discount_observed = np.maximum(pi_true + noise, -0.05)  # floor at -5% (small premium possible)

    df = pd.DataFrame({
        "trade_year": trade_year,
        "fund_type": fund_type,
        "L_t": L.round(3),
        "T_remaining": T_remaining.round(2),
        "pi_observed": discount_observed.round(4),
    })
    return df.sort_values(["trade_year", "L_t"]).reset_index(drop=True)


def generate_fund_cashflows(vintage_year=2014, fund_life=10,
                             commitment=100.0, seed=42):
    """Generate a single representative fund's cash flow schedule.

    Cash-flow pattern:
        - Investment period: years 0-5, gradual capital calls
        - Harvest period: years 4-10, distributions
        - Year 10: final distribution (residual NAV liquidation)

    Returns
    -------
    pd.DataFrame with columns ['date', 'year', 'cashflow', 'L_t', 'benchmark_index'].
    """
    rng = np.random.default_rng(seed)
    n_quarters = fund_life * 4
    dates = pd.date_range(start=f"{vintage_year}-01-01", periods=n_quarters, freq="QS")
    years_elapsed = np.arange(n_quarters) / 4.0

    # Capital calls
    call_intensity = np.exp(-(years_elapsed - 2.0) ** 2 / 2.0)  # peaks at year 2
    call_intensity[years_elapsed > 5] = 0  # no calls after year 5
    call_intensity = call_intensity / call_intensity.sum()
    total_called = commitment * 0.95  # 95% of commitment usually called
    calls = -call_intensity * total_called

    # Distributions
    dist_intensity = np.exp(-((years_elapsed - 7.0) ** 2) / 4.0)
    dist_intensity[years_elapsed < 3] = 0
    dist_intensity = dist_intensity / dist_intensity.sum()
    total_distributed = commitment * 1.8  # typical 1.8x TVPI
    dists = dist_intensity * total_distributed

    cashflows = calls + dists

    # L_t path: OU over the fund life, with some stress events injected
    L_t = np.zeros(n_quarters)
    L_t[0] = 0.5  # start in moderate boom
    for i in range(1, n_quarters):
        L_t[i] = L_t[i-1] + 0.45 * (1.0 - L_t[i-1]) * 0.25 + 0.32 * 0.5 * rng.standard_normal()

    # Benchmark (e.g., S&P TR) starts at 100 and grows
    bench_returns = rng.normal(0.08 * 0.25, 0.16 * np.sqrt(0.25), n_quarters)
    benchmark = 100 * np.cumprod(1 + bench_returns)

    return pd.DataFrame({
        "date": dates,
        "year": years_elapsed,
        "cashflow": cashflows.round(3),
        "L_t": L_t.round(3),
        "benchmark_index": benchmark.round(2),
    })


def write_all_synthetic_datasets(data_dir):
    """Generate and write all synthetic CSVs to disk."""
    import os
    os.makedirs(data_dir, exist_ok=True)

    L_hist = generate_L_history()
    L_hist.to_csv(f"{data_dir}/synthetic_L_history.csv", index=False)

    pricing = generate_synthetic_secondary_pricing()
    pricing.to_csv(f"{data_dir}/synthetic_secondary_pricing.csv", index=False)

    fund = generate_fund_cashflows()
    fund.to_csv(f"{data_dir}/sample_fund_cashflows.csv", index=False)

    return {
        "L_history": f"{data_dir}/synthetic_L_history.csv",
        "secondary_pricing": f"{data_dir}/synthetic_secondary_pricing.csv",
        "fund_cashflows": f"{data_dir}/sample_fund_cashflows.csv",
    }
