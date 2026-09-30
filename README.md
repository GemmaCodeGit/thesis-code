# Code for the doctoral thesis of Gemma Sedrakjan

Numerical companion to the thesis (TU Berlin, Institute of Mathematics). The
repository holds the code behind the numerical parts of two chapters:

| Folder | Chapter | Content |
|---|---|---|
| `games_with_signals/` | *Jump Signals in Optimal Investment* (with P. Bank) | Two-type mean field equilibrium with jump signals: best-response solver, fixed-point iteration, parameter sweeps and the certainty-equivalent comparisons of the section *Case Studies and Financial-Economic Discussion*. |
| `trading_with_the_flow/` | *Trading with the Flow* | The numerical scheme (Howard's policy iteration) for the quasi-variational inequality of the optimal execution problem, and the calibration of the resilience and volume-imbalance parameters to LOBSTER data (appendix *Parameter Calibration*). |

## Requirements

Python 3.10 or newer. Install the dependencies with

```bash
pip install -r requirements.txt
```

## `games_with_signals/games_with_signals.py`

One self-contained script. It defines the market and investor parameters,
the signal categories, the best-response objectives of the representative
investor (thesis eq. (mf_optimal_zero) and (mf_optimal_nonzero)), the
best-response iteration that computes a mean field equilibrium, and the
value constant `M` (thesis eq. (M_mf)) from which the certainty equivalent
`x_0^A = exp(M^{A,alt} - M^{A,ref})` is formed.

```bash
cd games_with_signals
python games_with_signals.py
```

solves the six scenarios of the thesis (relative performance concern
`theta` in {0.5, 1}, relative risk aversion `alpha` in {0.5, 2, 4}; base
parameters `p = ps = rho = 0.5`) into `scenario_outputs/theta_*_alpha_*/`.
Every parameter sweep is cached as an `.npz` file and skipped when the file
exists, so a run can be resumed. The sweeps use a process pool over all
cores; at the resolutions of the thesis (100 points per one-dimensional
sweep, 30 x 30 grids) one scenario takes several hours. The thesis results
were computed on the TU Berlin math cluster, one scenario per node.

Each `.npz` file stores the swept parameter values, the type environments,
the equilibrium strategy profiles and the certainty equivalents of type A;
the docstring of `run_all_experiments` lists which parameter each file
sweeps.

## `trading_with_the_flow/solver.py`

The finite-difference scheme for the QVI of the single-agent execution
problem on the grid inventory x bid liquidity x ask liquidity. The main
entry point is `run_qvi_pipeline(meta)`, which takes the experiment
description `meta` (grids, parameters, mark space, control sets) and runs
three cached stages:

1. `precompute_transitions` -- post-trade states and transition weights for
   every node, control and mark (parallel over inventory slices);
2. `solve_qvi` -- Howard's policy iteration backward in time;
3. `report` -- decision maps (optimal quotes, impulse sizes, impulse versus
   continuation advantages) for every time slice.

Each stage is stored as `data_{experiment}_{stage}.pkl.gz` in the working
directory and reused on the next call. The price-impact functions `I` and
`Xi` are the closed-form integrals of the impact kernel (via the
dilogarithm); `post_trade` implements the book mechanics for the agent's
impulses, exogenous market orders and exogenous limit orders.

## `trading_with_the_flow/lobster.ipynb`

Calibration notebook for the appendix *Parameter Calibration*:

* stationary law of the liquidity chain and the fixed point
  `lambda_r = mean of the stationary law` (Sections 1-2);
* empirical order-size distributions `P^1`, `P^2`, the scale `Q` and the
  capacity `lambda_max` from LOBSTER (Section 3, Tables `P1_emp`,
  `P2_emp`, `L32_stats`);
* Jensen-Shannon calibration of `(kappa, theta_f, theta_g)` (Sections 4-5,
  Table `table_fit`) and the benchmark choice (Section 6);
* conditional maximum likelihood for the volume-imbalance sensitivity
  `hat_kappa` (Section 7, Table `hat_kappa`).

**Data.** The notebook reads LOBSTER level-2 files for 2025-07-16, 10:30 to
15:00, for AAPL, AMAT, CSCO, EBAY, GOOG, INTC and MSFT. LOBSTER data is
licensed and not included. To re-run the data cells, place the files

```
trading_with_the_flow/lobster_data/{TICKER}_2025-07-16_34140000_57660000_message_2.csv
trading_with_the_flow/lobster_data/{TICKER}_2025-07-16_34140000_57660000_orderbook_2.csv
```

next to the notebook and start Jupyter from `trading_with_the_flow/`. The
model-only cells (stationary law, fixed point, benchmark parameters) run
without the data. Outputs have been cleared.
