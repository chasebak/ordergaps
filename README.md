# ordergaps

How much does the whole store sell? Exact order-number arithmetic + a temporal allocation model, validated against a simulator with ground truth.

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

![Weekly volume error falls as the observed fraction of orders grows](figures/hero_error_vs_fraction.png)

## Quickstart

```bash
git clone https://github.com/chasebak/ordergaps && cd ordergaps
python3 -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements-ci.txt
python -m pytest -q
python -m ordergaps.demo
```

Supported platforms: macOS arm64 and Linux. The demo simulates a store,
estimates its volume from one vendor's sales, and writes `out/report.md`
plus figures; it finishes in a few seconds.

## The idea

A marketplace numbers every order store-wide in one sequence. A vendor only
sees the orders containing its own items. The method is **exact
sequential-ID arithmetic + a temporal allocation model**: the gap between
two consecutively-seen order numbers is exactly how many store orders
happened in between — ID arithmetic, not estimation.

Saw order #1007 and then order #1012? Exactly four store orders happened in
between. The inference problem is WHEN those unseen orders happened and
through WHICH channel (in-store vs online/after-hours). The serial-number
idea goes back to WWII production
estimation (Ruggles & Brodie 1947; Goodman 1954), but unlike the German
tank problem, totals here are known exactly from ID differences — the open
question is timing and channel.

Caveat that applies to every number below: "volume" means orders CREATED,
not fulfilled — cancelled, voided and test orders consume order numbers and
are counted.

## How the estimator works

1. **Sessions** — store hours per weekday (closed days allowed), each day's
   session stretched to its first/last seen register (in-store) order; a
   closed day with a seen register order counts as open.
2. **Learned in-store pace** — orders per open minute per weekday, learned
   only from gaps that sit fully inside one session; pooled fallback for
   thin weekdays.
3. **Gap attribution** — for consecutive seen numbers n1 < n2, unseen =
   n2 − n1 − 1. The wall-clock span is split into open/closed minutes per
   day. Fully inside a session → all in-store. Fully outside → all
   online/after-hours. Mixed → expected in-store = Σ_day open_minutes ×
   rate(weekday), in-store = min(unseen, expected), online = remainder;
   in-store spread across days ∝ minutes × rate, online ∝ closed minutes.
4. **Day classification** — `full` (session ends pinned by seen orders, at
   least one inside), `spread` (zero seen orders — the day is a share of
   one long gap), `partial`, `closed`. Only full days enter per-day
   averages; spread days count in weekly totals only.
5. **Weekly tables** — total / in-store / online volume, vendor orders,
   vendor share of in-store, and vendor basket stats (orders, totals,
   items/order).
6. **Dispersion check** — uniform spreading within a segment is the
   conditional distribution of homogeneous-Poisson arrivals given the
   count; a Poisson vs negative-binomial check on full-day counts is
   computed and reported with every run.
7. **Pathology handling** — out-of-order timestamps (note + skip gap),
   numbering resets (detected, series split, affected weeks flagged),
   voided orders (consume numbers; see the caveat above).

The simulator (`ordergaps.simulator`) provides the ground truth the
production system never had: seeded nonhomogeneous arrivals (weekday ×
hour-of-day), store hours with closed days, two channels, a focal vendor
with configurable share, and planted pathologies — void orders, a numbering
reset, rate drift over weeks.

## Results (simulated ground truth, frozen seeds)

All numbers in this section are computed by this repo's code on the frozen
evaluation config; regenerate them with
`python -m ordergaps.evaluation.run_all --outdir figures`.

### 1. Error vs fraction of orders observed (hero figure above)

| vendor share | fraction observed | median weekly volume error |
|---|---|---|
| 0.02 | 0.020 | 0.3771% |
| 0.05 | 0.049 | 0.3048% |
| 0.10 | 0.097 | 0.1108% |
| 0.20 | 0.195 | 0.0676% |
| 0.35 | 0.344 | 0.0345% |
| 0.60 | 0.588 | 0.0150% |

Acceptance band: Spearman rho = -1.000 (required <= -0.8), inversions = 0
(required <= 1) -> PASS. Weekly totals are already sub-percent at a 2%
vendor share because interior counts are exact ID arithmetic; the residual
error lives at week boundaries and anomaly-flagged weeks.

### 2. Length bias of the learned weekday rates

The rate estimator learns only from gaps fully inside sessions, and the
observation times are the vendor's own sales — busy periods are
oversampled, so the learned pace is biased UP, and more so the less you
see. This is a known limitation, quantified rather than hidden:

| vendor share | mean bias | IQR |
|---|---|---|
| 0.02 | +6.03% | [+5.47%, +6.86%] |
| 0.05 | +4.57% | [+4.02%, +5.08%] |
| 0.10 | +2.72% | [+2.50%, +2.75%] |
| 0.20 | +1.65% | [+1.51%, +1.83%] |
| 0.35 | +1.06% | [+0.89%, +1.25%] |
| 0.60 | +0.76% | [+0.67%, +0.90%] |

![Length bias of learned rates](figures/length_bias.png)

### 3. Counted vs allocated (vendor share 0.10)

Error bars appear only where they belong — on the model-based rows:

| quantity | kind | error |
|---|---|---|
| window totals between seen orders | counted (ID arithmetic) | exact: 13713/13713 windows match ground truth |
| vendor share of orders, fully-pinned days | counted (sampling noise only) | mean abs deviation from true share: 0.0024 |
| per-day total volume, full days | counted (boundary spill only) | median rel err 0.82% |
| weekly in-store volume (channel split) | allocated (model) | median rel err 20.83% [IQR 20.15%, 21.07%] |
| per-day total volume, partial/spread days | allocated (timing model) | median rel err 0.86% [IQR 0.71%, 1.03%] |

The channel split is the weak joint, and the simulator says why: online
orders placed during open hours are attributed in-store by construction.
Totals stay near-exact; the split is honest model output.

## Production use

The same estimator tracked our share of the store's in-store orders growing
from 5% to 14% — production (private data).

## Design decisions / what didn't work

- **Spread days are excluded from per-day averages.** A spread day's count
  is just one long gap's model share — it carries no day-level signal, and
  averaging it in feeds the rate model its own output back.
- **The length-bias finding stays in.** First instinct was to "fix" the
  learned rates; the honest move was to quantify the bias (table above)
  and report it, since it is intrinsic to observing the sequence through
  the vendor's own sales.
- **Pin tolerance had to be relative.** The first rule pinned a session end
  only when ~1 expected order could hide past the boundary; at realistic
  shares the expected boundary wait is ~1/share orders, so almost no day
  ever classified as full. The shipped rule is max(5 orders, 5% of the
  day's expected total).
- **Timestamp precision bug.** Converting pandas Timestamps through
  `np.datetime64(ts)` silently truncates to microseconds and broke the
  window-exactness verification; `Timestamp.to_datetime64()` preserves
  nanoseconds. The exactness test exists to catch exactly this class of
  bug.
- **Simulator regimes were tuned on trial seeds, then frozen.** The
  committed figures and tables use seeds that were fixed before the
  committed results were generated, never adjusted after.

## What this does NOT claim

- No German-tank-problem optimality (no MVUE claims): totals come from ID
  differences, not serial-number extrapolation.
- Not fulfillment counts: "volume" = orders CREATED (voids, cancellations
  and test orders consume numbers and are counted).
- The channel split is model-based, not counted (median error ~20% in the
  simulated regime above); only totals and fully-pinned day counts are
  near-exact.
- Rates assume within-session homogeneity; the dispersion check (Poisson
  vs negative binomial on full-day counts) is computed and reported with
  every run, and the demo shows it flagging planted rate drift.
- Simulated results are statements about the simulated regimes, not about
  any particular real store.

## Repo map

- `src/ordergaps/` — estimator (`sessions`, `rates`, `attribution`,
  `classify`, `dispersion`, `estimator`), `simulator`, `evaluation/`,
  `demo`.
- `tests/` — gap-arithmetic exactness, session stretching, mixed-gap
  attribution, pathologies (out-of-order, reset, voids), dispersion check,
  evaluation reproducibility and the frozen acceptance band.
- `figures/` — committed PNGs generated by
  `python -m ordergaps.evaluation.run_all --outdir figures`.

Provenance: The estimator my business uses to size its market, rebuilt
against a simulator with ground truth.

References: Ruggles & Brodie (1947); Goodman (1954) — full citations and
the model write-up are in [MODEL.md](MODEL.md).

Part of a set of public repos extracted from my card-business tooling — see
my GitHub profile for the map.
