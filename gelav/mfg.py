"""Mean-field game fixed-point iteration for GE-LAV.

The MFG couples:
- HJB equation for V(L, t) (backward), given a population distribution mu(L, t).
- Fokker-Planck equation for mu(L, t) (forward), given a policy alpha*(L, t).

We iterate:

    given mu^(k):
        solve HJB -> V^(k+1), alpha*^(k+1)
        solve FP with alpha*^(k+1) -> mu^(k+1)
    until ||mu^(k+1) - mu^(k)|| < tol.

For pedagogical clarity, this implementation treats a simplified case
where the population coupling enters only through the term-structure
parameters alpha(mu), beta(mu), gamma(mu). Full MFG with state-feedback
through the exit policy is left as an exercise in the notebook.

Reference: Lasry-Lions (2007); Carmona-Delarue Vol. I.
"""
import numpy as np

from gelav.fokker_planck import solve_fp, stationary_density
from gelav.hjb import solve_hjb


def mfg_solver(
    kappa=0.45, sigma=0.32, L_bar=1.0,
    r=0.08, T_fund=10.0, NAV=1.0,
    coupling_strength=0.5,
    base_alpha=0.045, base_beta=-0.025, base_gamma=0.021,
    L_min=-3.0, L_max=3.0,
    n_L=101, n_t=100,
    max_iter=20, tol=1e-3,
    verbose=False,
):
    """Solve the simplified GE-LAV MFG by fixed-point iteration.

    Population coupling: pi-parameters depend on the cross-sectional mean E[L]:

        alpha_effective = base_alpha + coupling_strength * (L_bar - E[L])
        beta_effective  = base_beta
        gamma_effective = base_gamma

    (Higher distress in the population pushes alpha up; reflects the
    secondary-market crowding externality.)

    Returns
    -------
    dict with keys:
        'V', 'L_star', 't_grid', 'L_grid' from HJB at fixed point
        'mu' from FP at fixed point
        'iterations', 'converged', 'history'
    """
    L_grid = np.linspace(L_min, L_max, n_L)

    # Initialize mu to the stationary density
    mu_current = stationary_density(kappa, sigma, L_bar, L_grid)
    mu_current = mu_current / np.trapezoid(mu_current, L_grid)

    history = []
    converged = False
    hjb_result = None
    fp_result = None

    for iteration in range(max_iter):
        # Compute population mean
        E_L = float(np.trapezoid(L_grid * mu_current, L_grid))

        # Update effective pi-parameters
        alpha_eff = base_alpha + coupling_strength * (L_bar - E_L)
        beta_eff = base_beta
        gamma_eff = base_gamma

        # Effective pi function
        def pi_func(L, T_rem, a=alpha_eff, b=beta_eff, g=gamma_eff):
            L_arr = np.asarray(L, dtype=float)
            T_arr = np.asarray(T_rem, dtype=float)
            return (a + b * L_arr + g * L_arr ** 2) * np.sqrt(np.maximum(T_arr, 0))

        # Solve HJB with this pi
        hjb_result = solve_hjb(
            kappa=kappa, sigma=sigma, L_bar=L_bar,
            r=r, T_fund=T_fund, NAV=NAV,
            pi_func=pi_func,
            L_min=L_min, L_max=L_max,
            n_L=n_L, n_t=n_t,
        )

        # Solve FP forward (initialize with the previous mu)
        fp_result = solve_fp(
            initial_density=mu_current,
            kappa=kappa, sigma=sigma, L_bar=L_bar,
            T=T_fund, L_min=L_min, L_max=L_max,
            n_L=n_L, n_t=n_t,
        )
        mu_next = fp_result["p"][-1]  # final density

        # Check convergence
        diff = float(np.max(np.abs(mu_next - mu_current)))
        history.append({
            "iteration": iteration,
            "E_L": E_L,
            "alpha_eff": alpha_eff,
            "mu_diff": diff,
        })
        if verbose:
            print(f"  iter {iteration:2d}: E[L]={E_L:.4f}  "
                  f"alpha_eff={alpha_eff:.5f}  ||delta mu||={diff:.6f}")
        if diff < tol:
            converged = True
            mu_current = mu_next
            break
        mu_current = mu_next

    return {
        "V": hjb_result["V"] if hjb_result is not None else None,
        "L_star": hjb_result["L_star"] if hjb_result is not None else None,
        "t_grid": hjb_result["t_grid"] if hjb_result is not None else None,
        "L_grid": L_grid,
        "mu": mu_current,
        "iterations": iteration + 1,
        "converged": converged,
        "history": history,
        "final_alpha": history[-1]["alpha_eff"] if history else base_alpha,
    }
