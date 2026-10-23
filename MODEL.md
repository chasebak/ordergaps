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
