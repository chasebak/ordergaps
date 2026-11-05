### 4. Forecasting daily orders (rolling-origin backtest)

Setup: the frozen config above at vendor share 0.10; forecasting seeds 1101..1808 (pre-registered in PREDICTIONS.md); 28-day warm-up, 7-day horizon, an origin every 7 days: 6 origins per seed, 48 folds, 336 test days per series. At every origin the estimator is re-run on the orders seen before it, and the incomplete last day is dropped from both store histories. Regenerate with `python -m ordergaps.evaluation.forecast_experiment --outdir results` (writes results/, which this section copies; `--check` verifies the committed copy).

Headline: forecast error added by the partial view = +0.40% (weekday average: MAE 14.60 trained on the estimated history vs 14.54 trained on the true history), (a null: across the 8 seeds at this share it runs between -2.08% and +2.30%, sd 1.47%; across the 6 shares between -0.59% and +1.26%; a negative value means the estimated history did better than the truth by chance).

Owner view: next week's store orders, day by day, the weekday average's daily forecasts miss 5.64% (WAPE) vs 6.78% for last week's number; your own orders, 16.68% vs 21.77%.

#### 4.0 Forecast KPIs

| KPI | formula | why it matters | compared against | what the software does to it | measured on | value now |
|---|---|---|---|---|---|---|
| **Forecast error added by the partial view** (headline) | weekday-average MAE trained on re-estimated history ÷ the same trained on true history − 1 (same training and test days) | near 0 means one vendor can forecast the whole store from its own receipts | the same forecaster trained on full truth | rebuilds the history it learns from | synthetic truth | +0.40% (results/forecast_summary.json: headline.added) |
| Next-week orders, owner view | WAPE = Σ\|daily error\| ÷ Σ daily actual over the next 7 days' store orders (day by day, not the error on the week's sum) | "% of orders missed"; defined on zero days | seasonal naive | fewer orders missed | synthetic truth | 5.64% vs 6.78% (results/forecast_summary.json: owner) |
| Beats naive out of sample? | test MAE ÷ seasonal-naive test MAE; MASE beside it | does it beat what anyone would do by hand? | seasonal naive, same test days | below 1 means the software wins | synthetic truth | 0.832; MASE 0.823 (results/forecast_summary.json: tables.store_est.weekday_avg) |
| 80% interval coverage vs achievable | share of test days inside the 80% interval, beside the oracle's expected coverage | restock and cash need a range | the oracle's achievable coverage | shows whether the interval is honest | synthetic truth | 0.750 vs 0.811 (results/forecast_summary.json: tables.store_est.weekday_avg) |

Guarantees (tests, not KPIs): no look-ahead (changing orders after an origin leaves that fold unchanged; a run-once implementation fails the same test); full-view identity (share 1.0, no voids: the headline is exactly 0); poisson_quantile(10, 0.025) = 4 and poisson_quantile(10, 0.975) = 17; seasonal naive has zero error on an exactly weekly series.

#### 4.1 Store total, trained on the estimated history

| model | MAE mean ± sd | WAPE | MASE | beats naive? (MAE ratio) | 80% coverage | 95% coverage | achievable 80% / 95% |
|---|---|---|---|---|---|---|---|
| seasonal naive (same weekday last week) | 17.55 ± 6.08 | 6.78% | 0.990 | baseline (1.000) | 0.655 | 0.860 | 0.811 / 0.954 |
| 28-day average (context: weekday-blind) | 59.14 ± 3.81 | 22.84% | 3.304 | no (3.371) | 0.411 | 0.542 | 0.811 / 0.954 |
| weekday average (last 4) | 14.60 ± 4.27 | 5.64% | 0.823 | yes (0.832) | 0.750 | 0.908 | 0.811 / 0.954 |
| oracle: true expected count (floor) | 12.21 ± 3.66 | 4.72% | 0.688 | yes (0.696) | 0.827 | 0.961 | 0.811 / 0.954 |

> **Why this metric.** Owner's question: How many orders will the whole store take next week, from my own receipts?
> Kind of number: Forecast, scored out of sample against the simulator's true daily totals; the history it learns from is this repo's estimate, re-run at each origin.
> What would make it lie: Look-ahead (guarded by the poison test in tests/test_backtest.py) and Poisson arrivals by construction (no real-day shocks: see What this does NOT claim).

#### 4.2 Store total, trained on the true history

| model | MAE mean ± sd | WAPE | MASE | beats naive? (MAE ratio) | 80% coverage | 95% coverage | achievable 80% / 95% |
|---|---|---|---|---|---|---|---|
| seasonal naive (same weekday last week) | 16.85 ± 6.36 | 6.51% | 0.933 | baseline (1.000) | 0.655 | 0.866 | 0.811 / 0.954 |
| 28-day average (context: weekday-blind) | 59.15 ± 3.81 | 22.84% | 3.232 | no (3.510) | 0.411 | 0.542 | 0.811 / 0.954 |
| weekday average (last 4) | 14.54 ± 4.28 | 5.62% | 0.803 | yes (0.863) | 0.744 | 0.911 | 0.811 / 0.954 |
| oracle: true expected count (floor) | 12.21 ± 3.66 | 4.72% | 0.674 | yes (0.725) | 0.827 | 0.961 | 0.811 / 0.954 |

> **Why this metric.** Owner's question: How good could the same forecast be if I saw every store order?
> Kind of number: Forecast trained on the TRUE totals of exactly the same training days; it is the reference for the headline.
> What would make it lie: Unequal training days on the two sides (guarded by the matched-days and full-view identity tests in tests/test_backtest.py).

#### 4.3 Your own orders

| model | MAE mean ± sd | WAPE | MASE | beats naive? (MAE ratio) | 80% coverage | 95% coverage | achievable 80% / 95% |
|---|---|---|---|---|---|---|---|
| seasonal naive (same weekday last week) | 5.65 ± 1.50 | 21.77% | 1.023 | baseline (1.000) | 0.652 | 0.851 | 0.834 / 0.961 |
| 28-day average (context: weekday-blind) | 7.56 ± 1.37 | 29.14% | 1.370 | no (1.338) | 0.551 | 0.696 | 0.834 / 0.961 |
| weekday average (last 4) | 4.33 ± 1.07 | 16.68% | 0.787 | yes (0.766) | 0.795 | 0.943 | 0.834 / 0.961 |
| oracle: true expected count (floor) | 3.80 ± 1.09 | 14.65% | 0.689 | yes (0.673) | 0.851 | 0.970 | 0.834 / 0.961 |

> **Why this metric.** Owner's question: How many orders will I myself get next week?
> Kind of number: Forecast of the vendor's own daily orders, which are counted directly; truth = the simulator's realized vendor orders.
> What would make it lie: The simulated vendor also sees online orders (a real vendor sees register sales only); the oracle row shows how much is pure noise.

#### 4.4 Forecast error added by the partial view, by share

| vendor share | MAE trained on estimates | MAE trained on truth | error added |
|---|---|---|---|
| 0.02 | 14.43 | 14.51 | -0.59% |
| 0.05 | 14.73 | 14.54 | +1.26% |
| 0.10 | 14.60 | 14.54 | +0.40% |
| 0.20 | 14.47 | 14.54 | -0.47% |
| 0.35 | 14.52 | 14.54 | -0.13% |
| 0.60 | 14.51 | 14.54 | -0.22% |

Spread at vendor share 0.10: the same ratio pooled per seed runs between -2.08% and +2.30% across the 8 seeds (sd 1.47%), so the headline +0.40% is a null: the per-seed values straddle 0.

> **Why this metric.** Owner's question: Does seeing only part of the store cost me forecast accuracy?
> Kind of number: Ratio of two forecast MAEs on the same test days and training days, minus 1; near 0 means the partial view costs nothing.
> What would make it lie: Different training days (one side keeping the incomplete last day): the full-view identity test requires exactly 0 at share 1.0.

#### 4.5 Backtest checks (vendor share 0.10)

| check | value |
|---|---|
| training days a single run over all the data would rewrite | 464 of 2024 |
| learned weekday paces it would rewrite | 288 of 288 |
| dropped last day: estimated ÷ true orders | 0.9674 |
| training days per fold (mean); share of them full days | 42.17; 27.67% |
| test days with zero actual orders (all three series) | 0 |

> **Why this metric.** Owner's question: Is the backtest itself honest?
> Kind of number: Counted checks on the backtest's own inputs: what a single run over all the data would rewrite, and what the dropped last day held.
> What would make it lie: A run-once implementation; tests/test_backtest.py shows the poison test fails on one.

#### 4.6 Pre-registered predictions (PREDICTIONS.md, 2026-11-04)

- P1: weekday average MASE < 28-day average MASE (store, estimated history). Observed: 0.823 vs 3.304 -> PASS
- P2: weekday average test MAE / seasonal naive test MAE < 1. Observed: 0.832 -> PASS
- P3: forecast error added by the partial view < 5% at every share >= 0.05. Observed: 0.05: +1.26%; 0.10: +0.40%; 0.20: -0.47%; 0.35: -0.13%; 0.60: -0.22% -> PASS
- P4: seasonal naive 80% coverage <= achievable 80% coverage - 0.10. Observed: 0.655 vs achievable 0.811 -> PASS
- P5: oracle 80%/95% coverage within 3 Poisson-binomial SE of achievable. Observed: store_est 80%: 0.827 vs 0.811 (z +0.78); store_est 95%: 0.961 vs 0.954 (z +0.67); vendor 80%: 0.851 vs 0.834 (z +0.84); vendor 95%: 0.970 vs 0.961 (z +0.85) -> PASS
- B1: the oracle has the lowest pooled MAE in every series. Observed: store_est 12.21 vs best other 14.60; store_true 12.21 vs best other 14.54; vendor 3.80 vs best other 4.33 -> PASS
