# GE-LAV Companion Code

> Working code for the **GE-LAV** graduate finance course taught by Dr. Samir Asaf (PhD, CFA, CMA, CTP, CM&AA).
> Replicate every numerical result from the lecture decks in 8 Jupyter notebooks.

[![License: PolyForm Noncommercial 1.0.0](https://img.shields.io/badge/License-PolyForm%20Noncommercial%201.0.0-orange.svg)](https://polyformproject.org/licenses/noncommercial/1.0.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://github.com/drsamirasaf-creator/ge-lav-companion-code/actions/workflows/tests.yml/badge.svg)](https://github.com/drsamirasaf-creator/ge-lav-companion-code/actions/workflows/tests.yml)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/drsamirasaf-creator/ge-lav-companion-code/main)
[![Course Site](https://img.shields.io/badge/course-drsamirasaf--creator.github.io-1E3A5F.svg)](https://drsamirasaf-creator.github.io/ge-lav-course/)

The course (32 lecture sessions, 481 slides) covers a new framework for valuing
private capital that fixes three structural failures of DCF: liquidity illusion,
Jensen bias from convex discount factors, and partial-equilibrium treatment of
collective LP dynamics. **GE-LAV** = General Equilibrium Liquidity-Adjusted
Valuation, the synthesis presented in *Liquidity Illusion* (Asaf, 2026).

---

## What's in this repo

| Notebook | Topic | Sessions |
|---|---|---|
| `01_ou_calibration.ipynb` | Ornstein-Uhlenbeck process: simulate, calibrate (κ, σ, L̄) | S04, S19, S22 |
| `02_pi_term_structure.ipynb` | Fit π(L,T) to secondary market discount data | S05, S17 |
| `03_exit_boundary.ipynb` | Solve the HJB free-boundary problem for L*(t) | S11, S12, S20, S25 |
| `04_pme_la_irr_la_pme.ipynb` | Compute IRR / PME / LA-IRR / LA-PME for a sample fund | S07 |
| `05_mfg_solver.ipynb` | Mean-field game fixed-point iteration | S21, S26 |
| `06_jensen_bias.ipynb` | Quantify Jensen bias by asset class | S23, S29 |
| `07_stress_scenarios.ipynb` | Portfolio stress testing under L-shock paths | S18 |
| `08_pigouvian_tax.ipynb` | Compute τ*(L) and welfare gap closure | S24, S29 |

Each notebook is self-contained, runs in under 30 seconds on a laptop, and uses
**only synthetic data** (Lazard-like secondary pricing, sample fund cash flows,
30-year OU history). Drop in real data when you have it.

---

## Quick start (5 minutes)

### Option A — Clone and run locally

```bash
git clone https://github.com/drsamirasaf-creator/ge-lav-companion-code.git
cd ge-lav-companion-code
python -m venv .venv && source .venv/bin/activate
pip install -e .
jupyter notebook notebooks/
```

Tested with Python 3.10, 3.11, 3.12 on macOS, Linux, Windows.

### Option B — One-click in Google Colab

Each notebook has an **Open in Colab** badge at the top. Click it and Colab
will load the notebook with the `gelav` package auto-installed. No local
setup needed.

### Option C — Binder

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/drsamirasaf-creator/ge-lav-companion-code/main)

Launches a free, ephemeral JupyterLab in your browser with everything
pre-installed. Good for trying things out without committing to install.

---

## The `gelav` library

The notebooks import from a small Python package, `gelav`, that bundles all the
math primitives. You can use it standalone in your projects:

```python
from gelav.ou import simulate_ou, calibrate_ou
from gelav.hjb import solve_hjb, exit_boundary
from gelav.metrics import pme, la_pme
from gelav.term_structure import pi_of_L_T

# Simulate 1,000 OU paths over 10 years
paths = simulate_ou(L0=0.0, kappa=0.45, sigma=0.32, L_bar=1.0,
                    T=10.0, dt=1/252, n_paths=1000, seed=42)

# Solve for L*(t)
hjb = solve_hjb(kappa=0.45, sigma=0.32, L_bar=1.0, r=0.08, T_fund=10.0)
t_rem, L_star = exit_boundary(hjb)

# Compute LA-PME for a sample fund
adjusted = la_pme(cashflows, times, benchmark_index, L_path)
```

Module index:

| Module | Provides |
|---|---|
| `gelav.ou` | `simulate_ou`, `calibrate_ou`, `stationary_distribution`, `half_life` |
| `gelav.term_structure` | `pi_of_L`, `pi_of_L_T`, `fit_pi` |
| `gelav.metrics` | `irr`, `pme`, `la_irr`, `la_pme`, `tvpi`, `dpi` |
| `gelav.hjb` | `solve_hjb`, `exit_boundary` |
| `gelav.fokker_planck` | `solve_fp`, `stationary_density` |
| `gelav.mfg` | `mfg_solver` |
| `gelav.jensen` | `jensen_bias`, `jensen_bias_annualized`, `jensen_bias_exact` |
| `gelav.pigou` | `tau_star`, `welfare_gap_annual` |
| `gelav.data` | `generate_L_history`, `generate_synthetic_secondary_pricing` |

---

## Using your own data

The synthetic CSVs in `data/` mimic the structure of real datasets:

- `synthetic_L_history.csv` — quarterly L_t observations, 30 years.
  Real-world analog: Lazard Private Capital Advisory quarterly pricing reports,
  Greenhill / PJT / Evercore secondary market analyses.
- `synthetic_secondary_pricing.csv` — 500 secondary trades with (L, T, π).
  Real-world analog: any institutional secondary buyer's deal database.
- `sample_fund_cashflows.csv` — single fund's cash flows + L_t + benchmark.
  Real-world analog: GP-supplied LP quarterly statements + Bloomberg index.

To use your own data, simply replace the CSVs and re-run the notebook. Column
names are documented in `gelav/data.py`.

---

## For students

**Track 1 (practitioner):** Focus on notebooks 01, 02, 03, 04, 07. These cover
the empirical and decision-support tools you'll use in your IC memos and
portfolio analysis.

**Track 2 (researcher):** Focus on notebooks 03, 05, 06, 08. These cover the
numerical methods underlying the proofs (HJB, MFG fixed-point, Jensen, Pigou).

**Capstone project:** You're encouraged to clone this repo locally,
adapt the notebooks with your project's data, and submit your notebook as part
of the final paper. Code submissions get **+5% bonus** on the project grade
(see Session 30, slide 7). Per the license, classroom and individual research
use is permitted; redistribution outside the course requires written consent.

---

## For instructors using this repo

Each notebook ends with a **Suggested exercises** section that you can assign
as homework. The exercises are designed to be answerable with the existing
`gelav` API, requiring 5-20 lines of student-written code each.

---

## Citation

If you use this code in research or teaching, please cite:

> Asaf, S. (2026). *Liquidity Illusion*. (Manuscript in preparation.)
> Companion code: https://github.com/drsamirasaf-creator/ge-lav-companion-code

---

## License

**[PolyForm Noncommercial 1.0.0](LICENSE)** — this code is provided for
educational, classroom, and non-commercial research use only.

Permitted: classroom teaching, student coursework, academic research,
personal study, fair use under copyright law.

Not permitted without prior written consent: commercial use, SaaS deployment,
resale, incorporation into competing valuation products, paid consulting
deliverables, or use by for-profit organizations outside their internal
research and education functions.

The production GE-LAV(R) engine at https://liquidityillusion.com is
separately licensed and not covered by this repository.

For commercial licensing inquiries: drsamirasaf@gmail.com

## Contact

Dr. Samir Asaf — drsamirasaf@gmail.com — [course site](https://drsamirasaf-creator.github.io/ge-lav-course/)
