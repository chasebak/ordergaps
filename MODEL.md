# The ordergaps model

Notation: the vendor observes orders with store-wide sequential numbers.
For consecutive seen numbers n1 < n2 at times t1 <= t2, the unseen count in
the open interval is exactly u = n2 − n1 − 1. The estimator's job is to
place those u orders in time and channel.

## Sessions

Store hours are configured per weekday; a day may be closed. A day's
session is stretched to its first/last SEEN register (in-store) order (a
register order before open or after close proves the store was trading), and a
scheduled-closed day with seen register orders counts as open from its
first to its last register order. Online orders never stretch a session.

## Learned in-store pace

Rate(weekday) = orders per open minute, learned ONLY from gaps fully inside
one session: there the model attributes every unseen order to open-hours
trade, so (n2 − n1) arrivals over (t2 − t1) minutes estimates the pace
directly. Including the right endpoint's own arrival makes adjacent gaps
telescope: the gaps spanning one session sum to (n_last − n_first) orders
over (t_last − t_first) minutes. Weekdays with under `min_rate_minutes` of
fully-inside gap time fall back to the pooled rate; with no fully-inside
gaps at all, the pooled rate falls back to seen register orders over total
session minutes. The pooled fallback mixes weekdays of different scales —
it is total fully-inside count over total fully-inside minutes, a
minutes-weighted blend of busy and slow weekdays, so a thin weekday
inherits that blend rather than its own pace.

Length bias: the window endpoints are the vendor's own sales, which arrive
more often when the store is busy, so fully-inside windows oversample busy
stretches and the learned pace is biased up — by about +6% at a 2% vendor
share, shrinking toward zero as the share grows (quantified in README
experiment 2). The bias is intrinsic to the observation process, so it is
reported, not patched.

## Gap attribution

The gap's wall-clock span is split into per-day open and closed minutes
against the sessions. Three cases:

- fully inside sessions: all u orders are in-store;
- fully outside: all u orders are online/after-hours, spread across days
  proportionally to closed minutes;
- mixed: expected in-store E = Σ_day open_minutes(day) × rate(weekday),
  in-store = min(u, E), online = u − in-store; the in-store part is spread
  across days ∝ open_minutes × rate, the online part ∝ closed minutes.

Spreading uniformly in time within a segment is exactly the conditional
distribution of homogeneous-Poisson arrivals given the count; the
dispersion check below reports how well the homogeneity half of that
assumption holds.

## Day classification

- `closed`: no session.
- `spread`: a session but zero seen orders inside it — the day's volume is
  purely a share of one or more long gaps.
- `full`: at least one seen order inside the session and both session ends
  "pinned": the expected unseen count between the session boundary and the
  nearest inside seen order (boundary minutes × rate) is at most
  max(pin_tolerance_orders, pin_tolerance_fraction × the day's expected
  total). A full day's count is exact up to that boundary spill.
- `partial`: everything else.

Only full days enter per-day averages and the dispersion check, because
their counts are counted quantities; spread and partial days contribute to
weekly totals, where the ID arithmetic guarantees the sum.

## Dispersion check

On full-day in-store counts grouped by weekday (groups of at least 3 days):
the pooled conditional chi-square statistic T = Σ_w Σ_d (x_dw − x̄_w)² / x̄_w
with df = Σ_w (n_w − 1), whose null distribution under per-weekday Poisson
is approximately chi-square(df). The survival function is computed with a
regularized upper incomplete gamma (series + continued fraction; no SciPy
dependency). A method-of-moments negative binomial (Var = mu + alpha mu²,
alpha pooled across weekdays) provides the alternative log-likelihood. The
verdict is "overdispersed" only when p < 0.05, the dispersion index
exceeds 1, and the negative binomial log-likelihood beats Poisson.
Fractional full-day estimates are rounded to integers for the check.

## Pathologies

- Out-of-order timestamps (number increases, time regresses): the gap is
  skipped and its unseen count is reported as skipped, never guessed into
  the tables.
- Numbering resets: in time order, a number drop larger than
  `reset_min_drop` splits the series into eras; no gap crosses a reset. The
  destroyed span runs from the previous era's last observation to the first
  post-reset one (and often lies mostly in the ISO week BEFORE the reset),
  so every ISO week overlapping that span is flagged `has_anomaly` (the
  cross-reset count is unknowable).
- Voided/cancelled/test orders consume numbers and cannot be told apart
  from fulfilled ones inside a gap, hence the package-wide caveat: volume
  means orders CREATED, not fulfilled.

## Forecasting (rolling-origin backtest)

Target: orders per day (store total, and the vendor's own orders). A fold
with origin o trains on days before o and forecasts days o .. o + 6; origins
start after a 28-day warm-up and step by 7 days (`backtest.rolling_origins`).
At every origin `estimate()` is re-run on the vendor's orders created before
o, so no training value can depend on a later order; the estimator's last
seen day (incomplete: orders after the last sighting are not yet counted)
and any day inside a numbering-reset span are dropped, and the
"trained on truth" series uses exactly the same days.

Forecasts (`forecast.py`): seasonal naive ŷ_t = y_{t−7} (else y_{t−14}, …,
the latest same-weekday training day); the 28-day average (weekday-blind);
the weekday average of the last 4 same-weekday training values; and the
oracle μ_t = `simulator.true_daily_mean`, the expected count the Poisson
draws come from, scored at its median (the MAE-optimal point for a
Poisson count). Intervals are central Poisson intervals around the point
forecast, [q(α/2), q(1 − α/2)] with q the smallest k with CDF(k) ≥ p, all
quantiles from one cumulative-pmf loop. Because counts are whole numbers,
the oracle's 80% interval covers more than 80% (about 0.849 at μ = 10), so
coverage is judged against the oracle's exact expected coverage
("achievable"), with a Poisson-binomial standard error
sqrt(Σ p_i (1 − p_i)) / n.

Metrics: MAE; WAPE = Σ|e| ÷ Σ y; MASE = fold MAE ÷ in-sample seasonal-naive
MAE on the fold's training days; the out-of-sample ratio to seasonal naive
on the same test days; the headline = MAE(weekday average, estimated
history) ÷ MAE(weekday average, true history) − 1. Pre-registered
predictions: [PREDICTIONS.md](PREDICTIONS.md); choices and their
measurements: [DECISIONS.md](DECISIONS.md).

### Assumptions

| knob | value | source | what changes if wrong |
|---|---|---|---|
| warm-up | 28 days | design choice (4 same-weekday values for the weekday average) | shorter: the first folds' weekday averages rest on fewer values; longer: fewer origins |
| horizon / step | 7 days / 7 days | design choice ("next week's orders") | a longer horizon would widen errors under drift; overlapping test weeks would correlate folds |
| fold count | 6 origins per seed on the frozen 10-week config | design choice, provisional until the repo owner confirms it (keep one config for every README number) | more weeks would tighten the across-fold sd; the point metrics are pooled over 336 test days either way |
| weekday-average window | last 4 same-weekday values | design choice | a longer window lags drift more; a shorter one is noisier (variance μ/n) |
| 28-day average window | 28 values | design choice (a deliberately weekday-blind baseline) | none for the conclusion: it exists to show what ignoring the weekday costs |
| interval family | Poisson around the point forecast | design choice; the simulator's arrivals ARE Poisson | on real days with shocks (overdispersion) Poisson intervals under-cover; the oracle row would no longer be calibrated either |
| truth is Poisson around `true_daily_mean` | by construction | simulator design | tested (parity and calibration in tests/test_forecast.py); a simulator change that breaks it fails those tests |
| drift | +1% per week (linear, multiplicative) | illustrative (frozen config) | stronger drift hurts the 4-week weekday average first (it lags about 2.5 weeks of growth) |
| vendor share | 0.10 (main), 0.02..0.60 (headline by share) | illustrative (frozen config) | lower shares mean more spread days in the estimated history; the headline measures exactly that cost |
| what the vendor sees | register AND online orders | simulator design (README "Which data this uses") | a real vendor sees register sales only, so the "your own orders" series is easier here than in a real store |
| seeds | 1101..1808 | design choice, frozen before the first run | none by design; old seeds 101..808 are trial seeds for forecasting |

## Uniform-in-time baseline and full-day yield

The baseline (`baseline.py`) gets exactly the model's gaps, so exactly the
same unseen counts, and gives day d of a gap (t1, t2) with u unseen orders
u × (minutes of (t1, t2) on d) ÷ (minutes of (t1, t2)): no store hours, no
sessions, no learned pace. Because both allocators hand out every unseen
order once, a week's total = seen orders + the unseen orders of the gaps
inside it + a share of the gaps that straddle its start or end; a week no
gap with unseen orders straddles is identical under both (tested, with a
control in which straddled weeks do differ). README section 1.1 prints the
median weekly error of both on the same clean weeks, and how many weeks
differ at all.

Full-day yield = FULL days ÷ non-closed days over every labelled day,
pooled over seeds (README section 1.2). It counts the estimator's own
labels; whether a FULL label is earned is measured against truth in
section 3 (per-day error on full days).

### Assumptions

| knob | value | source | what changes if wrong |
|---|---|---|---|
| baseline spreading | even over wall-clock minutes of the gap | design choice (the simplest rival that uses the same counts) | a store-hours-aware baseline would sit closer to the model; this one shows what the timing model adds over the clock alone |
| tie rule | the two medians print the same at 4 decimals of a percent | design choice (tie only what the reader cannot tell apart) | a looser rule could print a baseline win as a tie; the "weeks whose totals differ" column shows when a tie is exact |
| "totals differ" tolerance | more than 1e-6 orders in a week | design choice (the same sums added in another order differ by far less) | a tolerance close to a real allocation difference would undercount the weeks that differ |
| yield denominator | every non-closed labelled day, first and last day included | design choice (one count over the estimator's day_labels) | covered days only would move each share's yield by at most the two edge days per seed |

## References

- Ruggles, R., & Brodie, H. (1947). An Empirical Approach to Economic
  Intelligence in World War II. Journal of the American Statistical
  Association, 42(237), 72-91.
- Goodman, L. A. (1954). Some Practical Techniques in Serial Number
  Analysis. Journal of the American Statistical Association, 49(265),
  97-112.

Unlike the German tank problem those papers formalize, totals here are
known exactly from ID differences; no minimum-variance-unbiased-estimator
machinery is used or claimed. The open inference problem — and everything
the simulator stress-tests — is timing and channel.
