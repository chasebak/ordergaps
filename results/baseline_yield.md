#### 1.1 Against a uniform-in-time baseline

The baseline takes the same gaps and the same unseen counts and spreads each gap evenly over its clock time, ignoring store hours, sessions and learned paces (`ordergaps.baseline`). Same seeds, same clean weeks. Regenerate with `python -m ordergaps.evaluation.run_all --outdir figures --baseline-outdir results`.

| vendor share | model (this repo) | uniform-in-time baseline | vs uniform (model - uniform) | clean weeks whose totals differ |
|---|---|---|---|---|
| 0.02 | 0.3771% | 0.5369% | -0.1598 pp: model lower | 29 of 48 |
| 0.05 | 0.3048% | 0.2563% | +0.0485 pp: uniform lower | 8 of 48 |
| 0.10 | 0.1108% | 0.1108% | 0.0000 pp: tie | 0 of 48 |
| 0.20 | 0.0676% | 0.0676% | 0.0000 pp: tie | 0 of 48 |
| 0.35 | 0.0345% | 0.0345% | 0.0000 pp: tie | 0 of 48 |
| 0.60 | 0.0150% | 0.0150% | 0.0000 pp: tie | 0 of 48 |

Totals are arithmetic; the model's value is the split and the day classes. Both allocators hand out the same unseen orders, so a week's total can differ only through a gap that straddles the week's boundary. Model lower at vendor share 0.02; tie at vendor share 0.10, 0.20, 0.35, 0.60; uniform lower at vendor share 0.05.

> **Why this metric.** Owner's question: Is the timing model worth anything for weekly totals, or would spreading each gap evenly over the clock do as well?
> Kind of number: Simulated: median |est - true| / true over clean weeks, the same weeks, gaps and unseen counts for both allocators; truth = the simulator's weekly totals.
> What would make it lie: A baseline fed different counts than the model (both must hand out the same total, and a week no gap straddles must come out identical: tests/test_baseline.py).

#### 1.2 Full-day yield

| vendor share | full days | non-closed days | full-day yield |
|---|---|---|---|
| 0.02 | 10 | 480 | 2.1% |
| 0.05 | 64 | 480 | 13.3% |
| 0.10 | 171 | 480 | 35.6% |
| 0.20 | 329 | 480 | 68.5% |
| 0.35 | 433 | 480 | 90.2% |
| 0.60 | 473 | 480 | 98.5% |

> **Why this metric.** Owner's question: How much of the store must I see before a single day's count means anything?
> Kind of number: Counted: the estimator's own day labels (FULL / non-closed days), pooled over seeds; no truth involved, so section 3's full-day error is what says FULL is earned.
> What would make it lie: A looser pin rule would raise the yield without making full days more exact (section 3 prints the per-day error on full days, measured against truth).

#### 1.3 Estimation KPIs

| KPI | formula | why it matters | compared against | what the software does to it | measured on | value now |
|---|---|---|---|---|---|---|
| Weekly volume error vs share | median \|est − true\| ÷ true over clean weeks | "How wrong am I at MY share?" | truth; the uniform-in-time baseline (section 1.1) | sub-percent at every share | synthetic truth (seeds 101..808) | model from 0.3771% at 0.02 to 0.0150% at 0.60; uniform from 0.5369% at 0.02 to 0.0150% at 0.60; model lower at vendor share 0.02; tie at vendor share 0.10, 0.20, 0.35, 0.60; uniform lower at vendor share 0.05 (results/baseline_yield.json: rows[*].model_weekly_err, rows[*].uniform_weekly_err) |
| Full-day yield | FULL days ÷ non-closed days, pooled over seeds | how much share you need before day-level numbers mean anything | across shares | rises with share | the estimator's own day labels (seeds 101..808) | from 2.1% at 0.02 to 98.5% at 0.60 (results/baseline_yield.json: rows[*].full_day_yield) |
| Channel-split error | median \|est in-store − true in-store\| ÷ true in-store, weekly | the in-store share is the number used with the store, so its error bar must hold up | truth | the model's weakest output, shown openly | synthetic truth (seeds 101..808) | 20.83% [IQR 20.15%, 21.07%] at vendor share 0.10 (results/baseline_yield.json: kpi.channel_err_median, kpi.channel_err_p25, kpi.channel_err_p75) |

Guarantees (tests, not KPIs): window exactness, 13713/13713 windows between seen orders match ground truth at vendor share 0.10 (results/baseline_yield.json: kpi.n_windows_exact, kpi.n_windows); tests/test_acceptance.py requires every window exact at every share and seed. It is exact by construction, so it is a test, not a KPI.
