"""Hamilton-Jacobi-Bellman PDE solver for the optimal exit problem.

The LP's value function V(L, t) solves, in the continuation region:

    dV/dt + kappa*(L_bar - L)*dV/dL + 0.5*sigma^2 * d2V/dL2
          - r * V + CF(L, t) = 0

with terminal condition V(L, T) = NAV * (1 - pi(L, 0)) and free boundary
L*(t) defined by value-matching + smooth-pasting:

    V(L*(t), t)        = NAV * (1 - pi(L*(t), T - t))         (value-match)
    dV/dL |_{L*(t)}    = -NAV * d/dL[pi(L*(t), T - t)]         (smooth-paste)

The numerical method is Crank-Nicolson with a penalty-style handling of the
free boundary at each time step (standard for American-option-like problems).

References
----------
- Wilmott, Howison, Dewynne, The Mathematics of Financial Derivatives, Ch 9.
- Book Chapters 20 and 25.
"""
import numpy as np
from scipy.sparse import diags, eye
from scipy.sparse.linalg import spsolve


def solve_hjb(
    kappa=0.45, sigma=0.32, L_bar=1.0,
    r=0.08,
    T_fund=10.0,
    NAV=1.0,
    cash_flow_func=None,
    pi_func=None,
    L_min=-3.0, L_max=3.0,
    n_L=201, n_t=200,
):
    """Solve the HJB free-boundary problem for V(L, t) and L*(t).

    Parameters
    ----------
    kappa, sigma, L_bar : float
        OU parameters.
    r : float
        Discount rate (annualized).
    T_fund : float
        Fund maturity in years.
    NAV : float
        Net asset value (numeraire; defaults to 1.0).
    cash_flow_func : callable or None
        f(L, t_remaining) -> instantaneous cash flow rate.
        Defaults to 0 (pure capital-gains play).
    pi_func : callable or None
        f(L, t_remaining) -> illiquidity premium fraction.
        Defaults to gelav.term_structure.pi_of_L_T.
    L_min, L_max : float
        Grid endpoints for L.
    n_L : int
        Number of L grid points.
    n_t : int
        Number of time steps.

    Returns
    -------
    dict with keys:
        'L_grid'     : np.ndarray, shape (n_L,)
        't_grid'     : np.ndarray, shape (n_t + 1,) -- decreasing from T to 0
        'V'          : np.ndarray, shape (n_t + 1, n_L) -- V(t_i, L_j)
        'L_star'     : np.ndarray, shape (n_t + 1,) -- exit boundary
        'parameters' : dict echoing inputs
    """
    if pi_func is None:
        from gelav.term_structure import pi_of_L_T
        pi_func = pi_of_L_T
    if cash_flow_func is None:
        cash_flow_func = lambda L, T_rem: 0.0

    L_grid = np.linspace(L_min, L_max, n_L)
    dL = L_grid[1] - L_grid[0]
    t_grid = np.linspace(T_fund, 0, n_t + 1)
    dt = abs(t_grid[1] - t_grid[0])

    # Coefficients of the spatial operator
    # A = kappa*(L_bar - L)*d/dL + 0.5*sigma^2*d2/dL2 - r*I
    mu = kappa * (L_bar - L_grid)             # advection coefficient at each node
    half_sig2 = 0.5 * sigma ** 2

    # Tridiagonal entries (interior nodes only; we'll handle BCs separately)
    # Central differences:
    # dV/dL  ~ (V_{j+1} - V_{j-1}) / (2*dL)
    # d2V/dL2 ~ (V_{j+1} - 2 V_j + V_{j-1}) / dL^2
    lower = mu[1:-1] * (-1 / (2 * dL)) + half_sig2 * (1 / dL ** 2)
    diag = -2 * half_sig2 / dL ** 2 - r
    upper = mu[1:-1] * (1 / (2 * dL)) + half_sig2 * (1 / dL ** 2)
    # Assemble sparse A for interior of size (n_L - 2)
    # Pad lower/upper to align with sparse diags spec
    A_interior = diags(
        [lower[1:], np.full(n_L - 2, diag), upper[:-1]],
        offsets=[-1, 0, 1],
        shape=(n_L - 2, n_L - 2),
        format="csc",
    )
    I = eye(n_L - 2, format="csc")

    # Crank-Nicolson matrices: (I - dt/2 * A) V_{n+1} = (I + dt/2 * A) V_n + dt * source
    M_lhs = I - (dt / 2) * A_interior
    M_rhs = I + (dt / 2) * A_interior

    # Storage
    V = np.empty((n_t + 1, n_L))
    L_star = np.empty(n_t + 1)

    # Terminal condition at t = T_fund: V(L, T) = NAV * (1 - pi(L, 0))
    payoff_terminal = NAV * (1 - pi_func(L_grid, 0.0))
    V[0] = payoff_terminal.copy()
    L_star[0] = L_bar  # at maturity, exit threshold collapses

    for n in range(n_t):
        t_now = t_grid[n]
        t_next = t_grid[n + 1]
        T_rem = T_fund - t_next  # time remaining from the perspective of t_next... wait
        # Actually t_grid is decreasing from T to 0. So at step n, we have V at t_now,
        # and we solve backward to t_next < t_now. Let's clarify "time remaining":
        # In the value function, T_remaining at time t_next is T_fund - t_next.
        # But our grid runs t_grid[0] = T_fund (terminal), going down. So time t_next
        # is t_grid[n+1], and time-remaining-until-T_fund seen from there is
        # T_fund - t_grid[n+1] = T_fund - t_next. That equals abs value of (T_fund - t_next).
        # We want pi_func(L, T_remaining_until_maturity_from_t_next).
        # Actually for the boundary at t_next, the exit payoff uses T - t_next = T_fund - t_next.
        # We're stepping V[n] (at t_now) -> V[n+1] (at t_next).
        # So we use T_rem corresponding to t_next.
        T_rem_for_boundary = T_fund - t_next

        # Source term: cash flow received during dt (treat as constant on interior, midpoint)
        # CF rate evaluated at L_grid, with T_rem at t_next:
        CF_vec = np.array([cash_flow_func(L, T_rem_for_boundary) for L in L_grid])
        source_interior = dt * CF_vec[1:-1]

        # Dirichlet BCs at L_min and L_max: hold V flat (or set to exit payoff)
        # For simplicity, set V at boundaries to the exit payoff at time t_next.
        exit_payoff = NAV * (1 - pi_func(L_grid, T_rem_for_boundary))
        V_left = exit_payoff[0]
        V_right = exit_payoff[-1]

        # Add boundary contributions to the RHS
        b = M_rhs @ V[n][1:-1] + source_interior
        # Boundary contribution at j=1 (from V_{j-1} = V_left) and j=n_L-2 (from V_{j+1} = V_right)
        # At j=1: -dt/2 * lower[0] * V_left from LHS  AND  +dt/2 * lower[0] * V_left from RHS
        # The LHS has (I - dt/2 A), so the off-diagonal term at j=1 contributes
        # -(-dt/2 * lower_coeff) * V_left = +dt/2 * lower_coeff * V_left to the RHS
        b[0] += (dt / 2) * (lower[0]) * V_left
        b[0] += (dt / 2) * (lower[0]) * V_left  # also from RHS shift
        b[-1] += (dt / 2) * (upper[-1]) * V_right
        b[-1] += (dt / 2) * (upper[-1]) * V_right

        # Solve interior
        V_next_interior = spsolve(M_lhs, b)

        # Reassemble full vector with Dirichlet boundaries
        V_next = np.empty(n_L)
        V_next[0] = V_left
        V_next[-1] = V_right
        V_next[1:-1] = V_next_interior

        # Enforce free-boundary: V(L, t) >= exit_payoff(L, T_rem)
        # If V_next < exit_payoff at any L, replace with exit_payoff (project).
        V_next = np.maximum(V_next, exit_payoff)
        V[n + 1] = V_next

        # Determine L*(t_next): smallest L for which V > exit_payoff (continuation begins)
        # Equivalently: largest L for which V == exit_payoff (in the stopping region below L_bar)
        # Stopping region is "low L" (distressed); above L*, continuation.
        is_stopping = np.isclose(V_next, exit_payoff, rtol=1e-6, atol=1e-9)
        # Find the rightmost stopping index, then L_star is the L right after that
        stop_idx = np.where(is_stopping)[0]
        if len(stop_idx) == 0:
            L_star[n + 1] = L_min  # no stopping region at this t
        else:
            # Take the largest index in the lower stopping region
            # (handle case where high-L region also matches due to Dirichlet BC)
            # We restrict to the region where L < L_bar to find the lower boundary
            low_region = stop_idx[L_grid[stop_idx] < L_bar]
            if len(low_region) == 0:
                L_star[n + 1] = L_min
            else:
                L_star[n + 1] = L_grid[low_region[-1]]

    return {
        "L_grid": L_grid,
        "t_grid": t_grid,
        "V": V,
        "L_star": L_star,
        "parameters": {
            "kappa": kappa, "sigma": sigma, "L_bar": L_bar,
            "r": r, "T_fund": T_fund, "NAV": NAV,
            "L_min": L_min, "L_max": L_max,
            "n_L": n_L, "n_t": n_t,
        },
    }


def exit_boundary(hjb_result):
    """Return (t_remaining, L*(t)) from the HJB solver output.

    Convenience accessor returning the exit boundary as a function of
    time remaining (T - t), which is the natural variable for plotting.
    """
    t_grid = hjb_result["t_grid"]
    T_fund = hjb_result["parameters"]["T_fund"]
    t_remaining = T_fund - t_grid  # increasing from 0 to T_fund
    return t_remaining, hjb_result["L_star"]
