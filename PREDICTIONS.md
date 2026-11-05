# Forecasting predictions (pre-registered)

Written 2026-11-04, BEFORE the first run of the forecast backtest on the
frozen forecasting seeds. Nothing below was changed after that run; the
results, whatever they are, are printed by
`python -m ordergaps.evaluation.forecast_experiment` (and by `run_all`) with
a PASS/FAIL line for each prediction. A FAIL ships as printed.

## Frozen setup

- **Simulator config:** the repo's frozen evaluation config, unchanged
  (`run_all.frozen_base_config()`: 10 weeks, vendor share 0.10, void rate
  0.02, numbering reset at the start of week 5, drift +1% per week). No new
  config is added for forecasting (Decision 17: keep the 10-week config and
  state its fold count).
- **Seeds:** 1101, 1202, 1303, 1404, 1505, 1606, 1707, 1808, frozen now.
  The old seeds 101..808 are trial seeds for forecasting (a throwaway
  prototype was looked at on them); they are never used for a forecast
  number in the README.
- **Folds (rolling origin, time-series cross-validation):** 28-day warm-up,
  7-day horizon, origins every 7 days. On 70 simulated days that is exactly
  **6 origins per seed** (days 28, 35, 42, 49, 56, 63), so **48 folds** and
  **336 test days** per series over the 8 seeds. Test days of different
  folds never overlap; training windows do (expanding window), so the
  across-fold sd is descriptive only.
- **Shares:** the main table uses vendor share 0.10. The headline is also
  computed at each of the repo's shares (0.02, 0.05, 0.10, 0.20, 0.35, 0.60)
  on the same seeds, for P3.

## Rules fixed before the run

- **No look-ahead:** at each origin, `estimate()` is re-run on the vendor's
  orders created before the origin only. A single run over all the data is
  not used anywhere (a test shows such a run changes past days).
- **Training days (matched):** the estimator's covered days at that origin
  (strictly after its first seen day and strictly before its last seen day,
  so the incomplete last open day before the origin is dropped), minus any
  day inside a numbering-reset span. The "trained on truth" series uses
  exactly the same days with their true totals, so the headline compares
  history quality only, not history length.
- **Your own orders** are fully known, so that series trains on every day
  before the origin.
- **Seasonal naive** ("same weekday last week"): the latest training value
  on the same weekday, i.e. t − 7; when t − 7 is not a training day (the
  dropped last day), t − 14, and so on back.
- **28-day average:** the mean of the last 28 training values (fewer when
  fewer exist); it ignores the weekday on purpose.
- **Weekday average:** the mean of the last 4 training values on the target
  day's weekday.
- **Oracle:** the simulator's true expected count for the day
  (`simulator.true_daily_mean`), scored at its Poisson median, which is the
  MAE-optimal point. It is the floor: what is left is pure Poisson noise.
- **Intervals:** every model's 80% and 95% interval is the central Poisson
  interval around its point forecast (the mean, before rounding):
  [q(α/2), q(1 − α/2)], q = the smallest k with Poisson CDF(k) ≥ p.
- **Metrics:** MAE per fold (mean and sd across folds, descriptive); WAPE =
  Σ|error| ÷ Σ actual over all test days (not MAPE: it reads as "% of
  orders missed" and stays defined on zero days); MASE = fold MAE ÷ the
  in-sample MAE of seasonal naive on that fold's training days, averaged
  across folds; "beats naive" = pooled test MAE ÷ seasonal-naive pooled test
  MAE on the same test days (< 1 = wins), reported beside MASE because a
  MASE below 1 alone does not prove a win out of sample; coverage = share of
  test days inside the interval; achievable = the oracle's expected coverage
  (the mean over test days of each day's exact Poisson coverage).
- **Structural zeros:** a day counts as a structural zero only when the
  series cannot have orders that day. Only the in-store series has them
  (closed Mondays). None of the three forecast series does: a closed Monday
  still has about 90 online orders, and the vendor sees online orders in the
  simulator. So every day is forecast and scored, closed Mondays included.
- **Headline:** forecast error added by the partial view = MAE of the
  weekday-average forecast trained on the estimated history ÷ MAE of the
  same forecast trained on the true history − 1; both scored on true daily
  totals over the same test days.

## Predictions

All on the store total trained on the estimated history (what a vendor
really has) at share 0.10, unless stated.

- **P1:** the weekday-aware forecast beats the weekday-blind one: weekday
  average MASE < 28-day average MASE.
- **P2:** the weekday average beats naive out of sample: its pooled test
  MAE ÷ seasonal naive's < 1. (Theory with no drift: error variance 1.25 μ
  vs 2 μ, ratio of MAEs about 0.79.)
- **P3:** the headline (forecast error added by the partial view) is under
  5% at every share ≥ 0.05. This is an expected null: the forecast is
  limited by Poisson noise, not by the gap estimator.
- **P4:** naive's Poisson interval under-covers: seasonal naive's empirical
  80% coverage is at least 0.10 below the achievable 80% coverage. (Theory:
  the point itself is one noisy draw, so the error variance is 2 μ while the
  interval assumes μ; about 0.65 vs about 0.83.)
- **P5:** the oracle matches its achievable coverage: its empirical 80% and
  95% coverage each lie within ±3 Poisson-binomial standard errors of the
  achievable coverage, on the store series and on your own orders. (The SE
  is Poisson-binomial because each day's achievable coverage differs.)
- **B1 (bound):** the oracle has the lowest pooled MAE of all models, in
  every series.
