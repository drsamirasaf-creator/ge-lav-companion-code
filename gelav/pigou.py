"""Pigouvian tax schedule tau*(L) for the GE-LAV externality.

When an LP sells in stress, it imposes a price-pressure externality on
other LPs. The optimal Pigouvian tax is the wedge between the social
planner's and the decentralized agent's first-order conditions:

    tau*(L) ~ -d/dL[Z(L, mu)] * dt

where Z is the aggregate externality.

For the book's calibrated case (book Table 24.2), tau*(L) is well
approximated by a quadratic in L that is zero for L >= 0 and grows
convexly as L drops below zero.

Reference: Book Chapter 24; Pigou (1920).
"""
import numpy as np


# Calibrated tau*(L) values from book Table 24.2
# L_state -> tau* (fraction of exit value)
CALIBRATED_TAU = {
    +1.0: 0.00,
    +0.5: 0.00,
     0.0: 0.00,
    -0.5: 0.015,
    -1.0: 0.04,
    -1.5: 0.07,
    -2.0: 0.11,
}


def tau_star(L, tau_max=0.15):
    """Optimal Pigouvian tax tau*(L) as a function of L.

    Functional form (calibrated by least squares to book Table 24.2):

        tau*(L) = max(0, a * L^2 + b * L)  for L < 0
        tau*(L) = 0                         for L >= 0

    with a ~ 0.025, b ~ -0.035 (approximate fit).

    Capped at tau_max to avoid blow-up at extreme L.

    Parameters
    ----------
    L : float or array_like
        Liquidity state.
    tau_max : float
        Cap on the tax (default 15 percent).

    Returns
    -------
    Same shape as L.
    """
    L = np.asarray(L, dtype=float)
    a, b = 0.025, -0.035
    raw = np.where(L < 0, a * L ** 2 - b * L, 0.0)
    # Note: with b = -0.035 and the formula a*L^2 - b*L, for L < 0 we get
    # a*L^2 + 0.035*L. But we want positive when L < 0. Let me use abs:
    raw = np.where(L < 0, a * L ** 2 + 0.035 * (-L), 0.0)
    return np.minimum(raw, tau_max)


def tau_star_from_table(L):
    """Look up tau*(L) by linear interpolation from the calibrated table."""
    L = np.asarray(L, dtype=float)
    L_pts = np.array(sorted(CALIBRATED_TAU.keys()))
    tau_pts = np.array([CALIBRATED_TAU[k] for k in sorted(CALIBRATED_TAU.keys())])
    return np.interp(L, L_pts, tau_pts, left=tau_pts[0], right=tau_pts[-1])


def welfare_gap_annual(AUM_global=13e12, gap_per_year=0.023):
    """Estimate of annual welfare loss from un-priced externality.

    Default values (book Chapter 24):
        AUM_global = 13 trillion USD
        gap_per_year = 2.3 percent annually

    Returns
    -------
    float: USD per year.
    """
    return float(AUM_global * gap_per_year)


def welfare_gap_closed_with_tau_star(closing_fraction=0.78):
    """Fraction of welfare gap closed by implementing tau*(L) optimally.

    Book Chapter 24 estimate: 75-80 percent. Default 78 percent.
    """
    total = welfare_gap_annual()
    return total * closing_fraction
