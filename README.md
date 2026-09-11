# GARCH(1,1) Volatility Modeling — From-Scratch Implementation

A from-first-principles implementation of the GARCH(1,1) conditional volatility
model, applied to real multi-asset ETF data, with honest out-of-sample
benchmarking against simpler volatility models.

## Motivation

Built while working as a quantitative intern analyzing market models under a
firm's CIO. The firm's existing risk model used a single, static expected-return
estimate (CAPM) per security. That approach hides how much *uncertainty*
actually surrounds any forecast. This project asks a more useful question:
**how does risk itself change over time, and does modeling that explicitly
actually improve on simpler approaches?**

## What's here

- **`garch.py`** — A GARCH(1,1) model class, fit via maximum likelihood
  estimation from scratch (no `arch` or `statsmodels` dependency — implemented
  directly from the model's mathematical definition using `scipy.optimize`).
- **`run_analysis.py`** — Applies the model to 3 years of daily returns across
  six ETFs spanning value, growth, core, bonds, and world equity, then:
  1. Fits GARCH(1,1) and reports parameters + diagnostics for each
  2. **Benchmarks GARCH against two simpler models** (constant variance,
     rolling 20-day realized variance) on held-out data — because a model is
     only worth using if it earns that complexity
  3. Simulates 1-year-ahead return distributions (5,000 paths per security)
  4. Produces volatility diagnostic plots

## Key finding — reported honestly, not cherry-picked

GARCH(1,1) **does not universally outperform simpler models.** In this
sample, it beat both a constant-variance model and a rolling-window model on
3 of 6 securities (VOE, AGG, VT), while a plain constant-variance model
actually fit better out-of-sample on 3 others (VTV, QQQ, VO).

This matters: it means GARCH's real edge — capturing that volatility clusters
after high-stress events, visible clearly in the diagnostic plots below,
where every equity ETF shows a sharp, shared spike around the same trading
day, likely a real market stress event — is genuine, but it isn't a free
lunch for every asset. A rigorous recommendation is asset-specific, not a
blanket "always use GARCH."

![Volatility plots](garch_volatility_plots.png)

## Model

```
r_t = mu + eps_t
eps_t = sigma_t * z_t,   z_t ~ N(0,1)
sigma_t^2 = omega + alpha * eps_(t-1)^2 + beta * sigma_(t-1)^2
```

Fit by maximizing the Gaussian log-likelihood via Nelder-Mead optimization,
subject to the stationarity constraint `alpha + beta < 1`.

## Sample results (3yr daily data, 6 representative ETFs)

| Ticker | Persistence (α+β) | Long-run Ann. Vol | Best model (OOS) |
|--------|---------------------|--------------------|-------------------|
| VTV    | 0.846               | 11.7%              | Constant Variance |
| QQQ    | 0.950               | 20.1%              | Constant Variance |
| VO     | 0.913               | 14.3%              | Constant Variance |
| VOE    | 0.837               | 13.2%              | **GARCH(1,1)**    |
| AGG    | 0.994               | 5.1%               | **GARCH(1,1)**    |
| VT     | 0.932               | 14.0%              | **GARCH(1,1)**    |

Note the bond ETF (AGG) shows the highest persistence (0.994) — volatility
shocks in bonds decay far more slowly than in equities, which matches
real-world fixed income behavior (duration and rate-sensitivity effects
linger) and is a genuine sanity check that the model is capturing something real.

## Why build this from scratch

The standard `arch` Python package wasn't available in the environment this
was built in. Rather than treat that as a blocker, implementing the
optimizer directly forces a full understanding of the model's mechanics —
useful both as a learning exercise and because it makes every assumption in
the pipeline auditable rather than hidden inside a library call.

## Usage

```python
from garch import GARCH11
import numpy as np

model = GARCH11().fit(daily_returns)
print(model.summary())

forecast = model.forecast(horizon_days=252, n_sims=5000)
print(f"Median 1yr forecast: {np.median(forecast):.1%}")
```

## Data

Daily return series derived from 3 years of ETF price history (VTV, QQQ, VO,
VOE, AGG, VT and others), originally sourced via a YCharts data feed for a
wealth management risk model.

## Author

Anaya Sethi — Statistics & Probability, Biology, UC San Diego
