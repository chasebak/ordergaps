# Decisions

Dated entries, one per choice, written as "decided X because measurement Y;
alternative Z lost by N". Nulls and reversals go in too. Every number below
is copied from `results/forecast.md` (README Results section 4),
`results/baseline_yield.md` (README sections 1.1 and 1.2), the Gap
Explorer's page data (`docs/data/*.json`) or from the pre-registration in
PREDICTIONS.md; `tests/test_decisions_numbers.py` fails if one is not.

## 2026-11-04 · Forecasting: rolling-origin folds, never shuffled k-fold

Decided time-series cross-validation (train on the past, forecast the next
7 days, step a week) because shuffling rows puts next week's orders into
the training set. Alternative "one estimate over all the data, then split"
lost by 464 of 2024 training days rewritten and 288 of 288 learned weekday
paces rewritten (section 4.5): with it, a training day's value depends on
orders that had not happened yet at the origin.

## 2026-11-04 · Re-run the estimator at every origin

Decided to re-run `estimate()` on the orders seen before each origin
because the measurement above shows a single run leaks the future into the
past. The poison test (tests/test_backtest.py) changes every order after an
origin and requires the fold's histories and forecasts to stay identical;
the run-once control fails the same check.

## 2026-11-04 · Drop the incomplete last day from BOTH store histories

Decided to drop the estimator's last seen day before each origin, on the
estimated and the true side alike, because that day is short: its estimate
held 0.9674 of its true orders (section 4.5), the orders after the last
sighting not being counted yet. Alternative "keep it on the estimated side
only" lost by breaking the full-view identity (the control in
tests/test_backtest.py): the headline would then mix estimation error with
a one-day difference in training data. Matched days make the headline
exactly 0 at share 1.0 with no voids.

## 2026-11-04 · Train on every covered day, not full days only

Decided to train the store forecasts on all of the estimator's covered days
because only 27.67% of the training days are full days at vendor share
0.10 (42.17 training days per fold on average, section 4.5). With all
covered days the partial view adds +0.40% to the weekday average's MAE
(headline). Alternative "full days only" was not run; it would train on
27.67% of the days, and its weekday mix would follow whichever days
happen to pin (not measured).

## 2026-11-04 · The headline is a null, and it ships as one

Decided to lead with "forecast error added by the partial view" because
it ties forecasting to the repo's core claim. Measured: +0.40% at share
0.10, and between -0.59% and +1.26% across all six shares (section 4.4);
P3 (under 5% at every share from 0.05 up) passes. Calling it a null is a
measurement too: pooled per seed at share 0.10, the same ratio runs between
-2.08% and +2.30% across the 8 seeds (sd 1.47%, section 4.4), so +0.40% sits
inside the seed-to-seed spread, and the negative values are seeds where the
estimated history did better than the truth by chance. The forecast is
limited by Poisson noise, not by the gap estimator.

## 2026-11-04 · WAPE, not MAPE

Decided WAPE (total miss ÷ total orders) because it reads as "% of orders
missed" and weights each day by its volume. Measured: 0 test days with
zero actual orders in the three series (section 4.5), so MAPE would be
defined here; WAPE is kept anyway because MAPE becomes undefined on the
in-store series' closed Mondays and over-weights slow days.

## 2026-11-04 · MASE beside an out-of-sample ratio to naive

Decided to report MASE (scaled by in-sample seasonal-naive MAE on the
training days) AND the test-MAE ratio to seasonal naive, because MASE
alone does not prove a win out of sample: seasonal naive itself scores a
MASE of 0.990 on the store history and 1.023 on your own orders (sections
4.1, 4.3), not 1. The weekday average beats naive on the same test days
(ratio 0.832, P2 PASS).

## 2026-11-04 · The weekday-blind baseline loses, as designed

Decided to keep the 28-day average as a context row, labelled "context:
weekday-blind" in tables 4.1, 4.2 and 4.3. Measured MASE 3.304
vs 0.823 for the weekday average (P1 PASS, section 4.1): that gap is
the price of ignoring the weekday profile. It is a straw man by design,
shown only as context.

## 2026-11-04 · Fold count: keep the frozen 10-week config (6 origins)

Provisional: the repo owner has not confirmed this choice yet. Decided to
keep the existing frozen config for forecasting, giving 6
origins per seed, 48 folds and 336 test days per series, because one
config then backs every README number and the frozen results stay
untouched. Measured: 336 test days are enough for the calibration check,
the oracle's 80% coverage is 0.827 vs 0.811 achievable (z +0.78, P5 PASS).
Alternative "a longer forecast-only config" was not built; it would mainly
tighten the across-fold sd, which is descriptive here.

## 2026-11-04 · Fresh seeds for forecasting

Decided to freeze 1101..1808 for forecasting before the first run, because
a throwaway prototype had already been looked at on 101..808, which are
now trial seeds. No logic or fixture test uses the frozen seeds, so no
test tunes on them; only the committed-output freshness check
(tests/test_forecast_outputs.py) re-runs two frozen (share, seed) slices to
compare them with results/.
Process note: the first run on the frozen seeds happened after
PREDICTIONS.md was written; later code changes were presentation only (a
regenerate sentence and a zero-day row in section 4), and every number
from that first run stayed the same.

## 2026-11-04 · The oracle is scored at its median

Decided to score the oracle's MAE at the Poisson median, the MAE-optimal
point for a count, because scoring the mean would hand the floor a small
handicap. Measured: the oracle has the lowest pooled MAE in every series
(B1 PASS; 12.21 vs 14.60 on the estimated-history store series).

## 2026-11-04 · Uniform-in-time baseline: print it, losses included

Decided to print a uniform-in-time baseline beside the model's weekly error
at every share, exactly as the code computes it, because the measurement is
mixed: the model is lower at vendor share 0.02 (0.3771% vs 0.5369%), the
baseline is lower at 0.05 (0.2563% vs 0.3048%, +0.0485 pp against the
model), and the two tie at 0.10 and above, where 0 of 48 clean weeks have
different totals at all (section 1.1). Alternative "drop the column, or tune
the pace model until it wins at 0.05" lost because tuning on the frozen
seeds would turn a printed loss into a fitted win; a printed loss is kept so
the honesty is visible. The reading the table supports is in
its own verdict line: totals are arithmetic; the model's value is the split
and the day classes.

## 2026-11-04 · Sections 1.1 and 1.2 sit after table 1, not inside it

Decided to put the uniform column and the full-day yield in their own
generated subsections right after section 1, instead of adding columns to
table 1, because the README's section 1 to 3 bytes are pinned by a hash
taken before any forecasting edit (tests/test_frozen_hashes.py). A new
column would force a re-pin, and a re-pinned hash no longer proves the
frozen numbers never moved. Section 1.1 repeats the model's error column so
the comparison still reads on one row; the hash test cuts the added block
out before hashing, and its control shows an edited section 1 number still
moves the hash.

## 2026-11-04 · Full-day yield: every labelled day, pooled over seeds

Decided to count the yield over every day the estimator labels, pooled
across seeds (FULL days ÷ non-closed days), because it is one count over
the estimator's own day labels, and the pooled counts print as plain
fractions: 10 of 480 at
vendor share 0.02, 171 of 480 at 0.10, 473 of 480 at 0.60 (section 1.2).
Alternative "covered days only" was not built: it moves the denominator by
at most the two edge days per seed and would stop matching the estimator's
own label table.

## 2026-11-04 · Gap Explorer: one interior week of one seed, at the frozen shares

Decided the page's week view shows seed 101's ISO 2025-W12 at each of the
frozen shares, because that week has no edge day and lies before the
planted numbering reset (tests/test_viz.py asserts both), and because the
seed's vendor orders are nested across shares, so the slider shows the
same week becoming countable rather than six different weeks. Measured:
the whole page bundle is 16949 + 25444 + 43257 bytes of data plus
meta.json (`docs/data/meta.json`), well under the 300000-byte budget.
Alternative "every seed's week" was not built: the tiles and tables are
already medians over all seeds, and a second week would show nothing the
tiles do not.

## 2026-11-04 · Worked gaps: the largest gap that starts each day, not every gap

Decided the page lists, per day of the week, only the gap with the most
unseen orders, because the number of gaps touching the week grows with the
share: 45 at vendor share 0.02 and 1021 at 0.60 (`docs/data/week.json`,
n_gaps_touching_week). Listing every gap would turn the page into an
order-level dump, which the bundle check forbids (no list as long as the
data rows). Every listed gap is checked against the simulator: unseen =
n2 - n1 - 1 = the orders created in between.

## 2026-11-04 · The page prints the README's own formatted text

Decided every tile and table on the page prints text formatted in Python by
the README's functions (run_all.counted_rows for section 3, the same
format strings for sections 1, 1.1 and 1.2), because the page should hold
no second rounding rule: the README's 13713/13713 and the page's
13,713 / 13,713 come from one call. Alternative "format raw numbers in
JavaScript" lost because a second formatter could round a printed loss into
a tie. The page's JavaScript still formats some values itself: chart axes
and tooltips, and the per-day and per-gap numbers of one week (week note,
week table, pace table, gap split) plus the hero-table and acceptance-note
values, each to a fixed number of decimals; the tiles and the section 3
table are Python text.

## 2026-11-04 · Why boxes under tables 1-3, cut out of the frozen hash

Decided to print the shared "why this metric" box under README tables 1, 2
and 3 (the same text the page shows) and to cut each box out before the
frozen README hash, because the hash pinned before any forecasting edit
must keep proving that the frozen numbers never moved; with the boxes cut,
it still matches. Alternative "re-pin the hash" lost because a re-pinned
hash proves nothing about the bytes it replaced. A box reworded in the
README only no longer cuts cleanly, so it moves the hash (control test).

## 2026-11-04 · The forecast view waits for v2

Decided to ship the page with the four estimation views and no forecast
view, because the estimation views explain the core claim on their own
and section 4's numbers are already printed in the README; a page view would need its own
data file and parity tests. Listed under "What to improve next".
