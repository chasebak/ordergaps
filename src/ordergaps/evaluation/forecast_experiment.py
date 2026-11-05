"""Forecast experiment: the rolling-origin backtest on the frozen config and
the frozen forecasting seeds, scored, summarized and checked against the
pre-registered predictions (PREDICTIONS.md).

    python -m ordergaps.evaluation.forecast_experiment --outdir results
    python -m ordergaps.evaluation.forecast_experiment --check

Writes results/forecast_folds.csv (one row per share, seed, origin, series,
model), results/forecast_origins.csv (one row per share, seed, origin),
results/forecast_summary.json and results/forecast.md (README Results
section 4, copied verbatim). --check recomputes everything and fails if the
committed files are stale. The full run takes about 25 s, so --check is a
local step before a commit; CI re-runs two (share, seed) slices and rebuilds
the summary and markdown from the committed CSVs
(tests/test_forecast_outputs.py).

Chain: tested code -> folds -> summary (computed from the folds read back
from the CSV, so the committed CSV alone reproduces it) -> markdown ->
README. Every forecast number in the README is copied from forecast.md.
"""

import argparse
import json
import math
import os
import sys
import time
from io import StringIO

import numpy as np
import pandas as pd

from ordergaps.backtest import (
    HORIZON_DAYS,
    MODELS,
    SERIES,
    STEP_DAYS,
    WARMUP_DAYS,
    build_fold,
    rolling_origins,
    score_fold,
)
from ordergaps.estimator import estimate
from ordergaps.evaluation.run_all import SHARES, frozen_base_config
from ordergaps.simulator import simulate, with_share

FORECAST_SEEDS = (1101, 1202, 1303, 1404, 1505, 1606, 1707, 1808)
MAIN_SHARE = 0.10
HEADLINE_SERIES = ("store_est", "store_true")
HEADLINE_MODEL = "weekday_avg"
SIGNIFICANT_DIGITS = 10
DEFAULT_OUTDIR = "results"
FOLDS_FILE = "forecast_folds.csv"
ORIGINS_FILE = "forecast_origins.csv"
SUMMARY_FILE = "forecast_summary.json"
MARKDOWN_FILE = "forecast.md"

MODEL_LABELS = {
    "naive": "seasonal naive (same weekday last week)",
    "avg28": "28-day average (context: weekday-blind)",
    "weekday_avg": "weekday average (last 4)",
    "oracle": "oracle: true expected count (floor)",
}
SERIES_TITLES = {
    "store_est": "4.1 Store total, trained on the estimated history",
    "store_true": "4.2 Store total, trained on the true history",
    "vendor": "4.3 Your own orders",
}

# The "why this metric" box: one copy, written into forecast_summary.json and
# forecast.md, which the README copies verbatim (tests/test_readme_drift.py).
WHY_THIS_METRIC = {
    "store_est": {
        "question": "How many orders will the whole store take next week, from my own receipts?",
        "kind": (
            "Forecast, scored out of sample against the simulator's true daily totals; "
            "the history it learns from is this repo's estimate, re-run at each origin."
        ),
        "lie": (
            "Look-ahead (guarded by the poison test in tests/test_backtest.py) and Poisson "
            "arrivals by construction (no real-day shocks: see What this does NOT claim)."
        ),
    },
    "store_true": {
        "question": "How good could the same forecast be if I saw every store order?",
        "kind": (
            "Forecast trained on the TRUE totals of exactly the same training days; "
            "it is the reference for the headline."
        ),
        "lie": (
            "Unequal training days on the two sides (guarded by the matched-days and "
            "full-view identity tests in tests/test_backtest.py)."
        ),
    },
    "vendor": {
        "question": "How many orders will I myself get next week?",
        "kind": (
            "Forecast of the vendor's own daily orders, which are counted directly; "
            "truth = the simulator's realized vendor orders."
        ),
        "lie": (
            "The simulated vendor also sees online orders (a real vendor sees register "
            "sales only); the oracle row shows how much is pure noise."
        ),
    },
    "by_share": {
        "question": "Does seeing only part of the store cost me forecast accuracy?",
        "kind": (
            "Ratio of two forecast MAEs on the same test days and training days, minus 1; "
            "near 0 means the partial view costs nothing."
        ),
        "lie": (
            "Different training days (one side keeping the incomplete last day): the "
            "full-view identity test requires exactly 0 at share 1.0."
        ),
    },
    "diagnostics": {
        "question": "Is the backtest itself honest?",
        "kind": (
            "Counted checks on the backtest's own inputs: what a single run over all the "
            "data would rewrite, and what the dropped last day held."
        ),
        "lie": (
            "A run-once implementation; tests/test_backtest.py shows the poison test "
            "fails on one."
        ),
    },
}


def round_significant(value, digits=SIGNIFICANT_DIGITS):
    """Round a float to `digits` significant digits (ints and NaN unchanged)."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    number = float(value)
    if math.isnan(number) or number == 0.0:
        return number
    return float(f"{number:.{digits}g}")


def round_structure(value):
    """round_significant applied through dicts and lists (for JSON)."""
    if isinstance(value, dict):
        rounded = {}
        items = value.items()
        for key, item in items:
            rounded[key] = round_structure(item)
        return rounded
    if isinstance(value, (list, tuple)):
        rounded_list = []
        for item in value:
            rounded_list.append(round_structure(item))
        return rounded_list
    if isinstance(value, (float, int, np.floating, np.integer)):
        return round_significant(value)
    return value


def leak_check(sim, origins):
    """What a single estimate() over ALL the data would rewrite in the past.

    For each origin: compare the re-estimated training values (the honest
    history) with the run-once estimate's values on the same days, and the
    learned weekday paces. Counts days / paces that differ (|diff| > 1e-6).
    Baseline: the honest per-origin re-estimate (build_fold).
    Failure mode: the 1e-6 tolerance (1e-12 on paces) would hide a rewrite
    smaller than a millionth of an order, so "0 changed" means "nothing
    above the tolerance", not bit-identical.
    """
    config = sim.config
    start_date = pd.Timestamp(config.start_day).date()
    run_once = estimate_all(sim)
    run_once_values = {}
    daily_tuples = list(run_once.daily.itertuples(index=False))
    for row in daily_tuples:
        run_once_values[(row.day - start_date).days] = float(row.est_total)
    rows = []
    for origin in origins:
        fold = build_fold(sim, origin)
        history = fold.histories["store_est"]
        days_compared = 0
        days_changed = 0
        history_items = sorted(history.items())
        for day, value in history_items:
            days_compared += 1
            if abs(run_once_values[day] - value) > 1e-6:
                days_changed += 1
        rates_compared = 0
        rates_changed = 0
        weekday_values = range(7)
        for weekday in weekday_values:
            honest_rate = fold.learned_rates.get(weekday)
            once_rate = run_once.rates.per_weekday.get(weekday)
            if honest_rate is None and once_rate is None:
                continue
            rates_compared += 1
            one_missing = honest_rate is None or once_rate is None
            if one_missing or abs(honest_rate - once_rate) > 1e-12:
                rates_changed += 1
        rows.append(
            {
                "origin": origin,
                "leak_days_compared": days_compared,
                "leak_days_changed": days_changed,
                "leak_rates_compared": rates_compared,
                "leak_rates_changed": rates_changed,
            }
        )
    return rows


def estimate_all(sim):
    """The run-once estimate (used ONLY to measure the leak it would cause)."""
    return estimate(sim.observed, sim.config.store_hours)


def keep_fold_row(share, row):
    """All rows at the main share; the headline pair only at other shares."""
    if share == MAIN_SHARE:
        return True
    return row["series"] in HEADLINE_SERIES and row["model"] == HEADLINE_MODEL


def run_forecast_experiment(shares=SHARES, seeds=FORECAST_SEEDS):
    """(folds, origins) DataFrames for every share and seed.

    folds: one score row per (share, seed, origin, series, model), every row
    at MAIN_SHARE and only the headline pair elsewhere (keep_fold_row).
    origins: one row per (share, seed, origin) with the training-day counts
    and the dropped last day; the leak columns are measured at MAIN_SHARE
    only and hold -1 ("not measured") at other shares.
    """
    base = frozen_base_config()
    n_days = base.n_weeks * 7
    origins = rolling_origins(n_days, WARMUP_DAYS, HORIZON_DAYS, STEP_DAYS)
    fold_rows = []
    origin_rows = []
    for share in shares:
        config = with_share(base, share)
        for seed in seeds:
            sim = simulate(config, seed)
            leak_by_origin = {}
            if share == MAIN_SHARE:
                leak_rows = leak_check(sim, origins)
                for leak_row in leak_rows:
                    leak_by_origin[leak_row["origin"]] = leak_row
            for origin in origins:
                fold = build_fold(sim, origin)
                scored = score_fold(fold)
                for row in scored:
                    if keep_fold_row(share, row):
                        fold_rows.append(row)
                origin_row = {
                    "share": share,
                    "seed": seed,
                    "origin": origin,
                    "n_train_days": len(fold.histories["store_est"]),
                    "n_full_train_days": fold.n_full_train_days,
                    "n_reset_span_days": fold.n_reset_span_days,
                    "dropped_day": fold.dropped_day,
                    "dropped_est": fold.dropped_est,
                    "dropped_true": fold.dropped_true,
                }
                leak_row = leak_by_origin.get(origin)
                leak_keys = (
                    "leak_days_compared",
                    "leak_days_changed",
                    "leak_rates_compared",
                    "leak_rates_changed",
                )
                for key in leak_keys:
                    if leak_row is None:
                        origin_row[key] = -1
                    else:
                        origin_row[key] = leak_row[key]
                origin_rows.append(origin_row)
    folds = pd.DataFrame(fold_rows)
    origins_frame = pd.DataFrame(origin_rows)
    return folds, origins_frame


def frame_to_csv_text(frame):
    """CSV with floats rounded to 10 significant digits (stable text)."""
    rounded = frame.copy()
    columns = list(rounded.columns)
    for column in columns:
        if rounded[column].dtype.kind == "f":
            values = list(rounded[column])
            new_values = []
            for value in values:
                new_values.append(round_significant(value))
            rounded[column] = new_values
    buffer = StringIO()
    rounded.to_csv(buffer, index=False, lineterminator="\n")
    return buffer.getvalue()


def csv_text_to_frame(text):
    return pd.read_csv(StringIO(text))


def _pooled(rows, column):
    return float(rows[column].sum())


def summarize_series_model(rows):
    """Pooled metrics for one (series, model) over all its folds.

    MAE mean/sd across folds (sd descriptive: training windows overlap);
    WAPE = sum|err| / sum actual; MASE = mean of fold MASEs; coverage =
    hits / test days; achievable = mean exact oracle coverage; se = the
    Poisson-binomial SE sqrt(sum p(1 - p)) / n. Baseline: the caller adds
    ratio_vs_naive (pooled MAE / seasonal naive's pooled MAE).
    Failure mode: MASE is the mean of fold MASEs, so one fold with a tiny
    in-sample naive scale can dominate it; the pooled MAE ratio beside it
    does not have that weakness.
    """
    n_days = _pooled(rows, "n_days")
    fold_maes = rows["mae"].to_numpy(dtype=float)
    if len(fold_maes) > 1:
        mae_sd = float(np.std(fold_maes, ddof=1))
    else:
        mae_sd = math.nan
    summary = {
        "n_folds": len(rows),
        "n_days": int(n_days),
        "mae": _pooled(rows, "sum_abs_err") / n_days,
        "mae_mean": float(np.mean(fold_maes)),
        "mae_sd": mae_sd,
        "wape": _pooled(rows, "sum_abs_err") / _pooled(rows, "sum_actual"),
        "mase": float(np.mean(rows["mase"].to_numpy(dtype=float))),
        "cov80": _pooled(rows, "n_in80") / n_days,
        "cov95": _pooled(rows, "n_in95") / n_days,
        "width80": _pooled(rows, "sum_width80") / n_days,
        "achievable80": _pooled(rows, "achievable80_sum") / n_days,
        "achievable95": _pooled(rows, "achievable95_sum") / n_days,
        "se80": math.sqrt(_pooled(rows, "achievable80_var")) / n_days,
        "se95": math.sqrt(_pooled(rows, "achievable95_var")) / n_days,
    }
    return summary


def headline_spread(folds, by_share):
    """How far the headline moves across seeds and shares (is it a null?).

    Per seed at MAIN_SHARE: the same ratio as the headline, pooled over that
    seed's folds only (weekday-average MAE trained on estimates / trained on
    truth - 1, same training and test days). Returns its min, max and sd
    across seeds, plus the min and max of by_share's "added". Rule: the
    headline is called a null when the per-seed values straddle 0 (some
    seeds' estimated history beat the truth, some lost to it).
    Baseline: 0, the value a partial view that costs nothing would give.
    Failure mode: 8 seeds is a small sample, so the sd is descriptive (the
    seeds' folds share no data, but each seed's 6 origins overlap); a value
    inside the per-seed range is called a null, not proven to be one.
    """
    main_rows = folds[folds["share"] == MAIN_SHARE]
    headline_rows = main_rows[main_rows["model"] == HEADLINE_MODEL]
    seed_values = sorted(headline_rows["seed"].unique())
    seed_added = []
    for seed in seed_values:
        seed_rows = headline_rows[headline_rows["seed"] == seed]
        est_rows = seed_rows[seed_rows["series"] == "store_est"]
        true_rows = seed_rows[seed_rows["series"] == "store_true"]
        seed_mae_est = _pooled(est_rows, "sum_abs_err") / _pooled(est_rows, "n_days")
        seed_mae_true = _pooled(true_rows, "sum_abs_err") / _pooled(true_rows, "n_days")
        seed_added.append(seed_mae_est / seed_mae_true - 1.0)
    share_added = []
    for entry in by_share:
        share_added.append(entry["added"])
    if len(seed_added) > 1:
        seed_sd = float(np.std(seed_added, ddof=1))
    else:
        seed_sd = math.nan
    return {
        "n_seeds": len(seed_added),
        "n_shares": len(share_added),
        "seed_added_min": float(min(seed_added)),
        "seed_added_max": float(max(seed_added)),
        "seed_added_sd": seed_sd,
        "share_added_min": float(min(share_added)),
        "share_added_max": float(max(share_added)),
        "is_null": bool(min(seed_added) < 0.0 < max(seed_added)),
    }


def null_word(headline):
    if headline["is_null"]:
        return "a null"
    return "not a null"


def null_clause(headline):
    """'(a null: ...)' after the headline value, every number from the summary."""
    return (
        f"({null_word(headline)}: across the {headline['n_seeds']} seeds at this share it runs between "
        f"{headline['seed_added_min']:+.2%} and {headline['seed_added_max']:+.2%}, sd "
        f"{headline['seed_added_sd']:.2%}; across the {headline['n_shares']} shares between "
        f"{headline['share_added_min']:+.2%} and {headline['share_added_max']:+.2%}; a "
        "negative value means the estimated history did better than the truth by chance)"
    )


def summarize(folds, origins):
    """The summary dict behind forecast.md and the README lines."""
    tables = {}
    main_rows = folds[folds["share"] == MAIN_SHARE]
    for series in SERIES:
        series_table = {}
        for model in MODELS:
            mask = (main_rows["series"] == series) & (main_rows["model"] == model)
            series_table[model] = summarize_series_model(main_rows[mask])
        naive_mae = series_table["naive"]["mae"]
        for model in MODELS:
            ratio = series_table[model]["mae"] / naive_mae
            series_table[model]["ratio_vs_naive"] = ratio
        tables[series] = series_table

    by_share = []
    share_values = sorted(folds["share"].unique())
    for share in share_values:
        share_rows = folds[(folds["share"] == share) & (folds["model"] == HEADLINE_MODEL)]
        est_rows = share_rows[share_rows["series"] == "store_est"]
        true_rows = share_rows[share_rows["series"] == "store_true"]
        mae_est = _pooled(est_rows, "sum_abs_err") / _pooled(est_rows, "n_days")
        mae_true = _pooled(true_rows, "sum_abs_err") / _pooled(true_rows, "n_days")
        by_share.append(
            {
                "share": float(share),
                "mae_est": mae_est,
                "mae_true": mae_true,
                "added": mae_est / mae_true - 1.0,
            }
        )
    headline = None
    for entry in by_share:
        if entry["share"] == MAIN_SHARE:
            headline = dict(entry)
    spread = headline_spread(folds, by_share)
    headline.update(spread)

    owner = {
        "store_wape_weekday": tables["store_est"]["weekday_avg"]["wape"],
        "store_wape_naive": tables["store_est"]["naive"]["wape"],
        "vendor_wape_weekday": tables["vendor"]["weekday_avg"]["wape"],
        "vendor_wape_naive": tables["vendor"]["naive"]["wape"],
    }

    main_origins = origins[origins["share"] == MAIN_SHARE]
    zero_days = 0
    for series in SERIES:
        oracle_mask = (main_rows["series"] == series) & (main_rows["model"] == "oracle")
        oracle_rows = main_rows[oracle_mask]
        zero_days += int(oracle_rows["n_zero_actual"].sum())
    diagnostics = {
        "leak_days_changed": int(main_origins["leak_days_changed"].sum()),
        "leak_days_compared": int(main_origins["leak_days_compared"].sum()),
        "leak_rates_changed": int(main_origins["leak_rates_changed"].sum()),
        "leak_rates_compared": int(main_origins["leak_rates_compared"].sum()),
        "dropped_est_sum": float(main_origins["dropped_est"].sum()),
        "dropped_true_sum": float(main_origins["dropped_true"].sum()),
        "dropped_ratio": float(
            main_origins["dropped_est"].sum() / main_origins["dropped_true"].sum()
        ),
        "train_days_mean": float(main_origins["n_train_days"].mean()),
        "full_train_fraction": float(
            main_origins["n_full_train_days"].sum() / main_origins["n_train_days"].sum()
        ),
        "zero_actual_test_days": zero_days,
    }

    n_origins = int(main_origins["origin"].nunique())
    n_seeds = int(main_origins["seed"].nunique())
    setup = {
        "seeds": list(FORECAST_SEEDS),
        "main_share": MAIN_SHARE,
        "shares": list(SHARES),
        "n_weeks": frozen_base_config().n_weeks,
        "warmup_days": WARMUP_DAYS,
        "horizon_days": HORIZON_DAYS,
        "step_days": STEP_DAYS,
        "n_origins_per_seed": n_origins,
        "n_folds": n_origins * n_seeds,
        "n_test_days": tables["store_est"]["naive"]["n_days"],
    }
    summary = {
        "setup": setup,
        "tables": tables,
        "by_share": by_share,
        "headline": headline,
        "owner": owner,
        "diagnostics": diagnostics,
        "why_this_metric": WHY_THIS_METRIC,
    }
    summary["predictions"] = prediction_verdicts(summary)
    return summary


def prediction_verdicts(summary):
    """P1-P5 and B1 exactly as worded in PREDICTIONS.md (2026-11-04).

    Rule: each verdict compares the summary numbers with the threshold the
    pre-registration fixed (P3 < 5% added at every share >= 0.05, P4 naive
    80% coverage at least 0.10 below achievable, P5 |z| <= 3, ...).
    Failure mode: the verdicts are only as frozen as PREDICTIONS.md's
    wording; editing a threshold here and there together would pass
    silently, so the claims are printed beside each verdict.
    """
    tables = summary["tables"]
    store = tables["store_est"]
    verdicts = []

    p1_weekday = store["weekday_avg"]["mase"]
    p1_avg28 = store["avg28"]["mase"]
    verdicts.append(
        {
            "id": "P1",
            "claim": "weekday average MASE < 28-day average MASE (store, estimated history)",
            "observed": f"{p1_weekday:.3f} vs {p1_avg28:.3f}",
            "passed": bool(p1_weekday < p1_avg28),
        }
    )

    p2_ratio = store["weekday_avg"]["ratio_vs_naive"]
    verdicts.append(
        {
            "id": "P2",
            "claim": "weekday average test MAE / seasonal naive test MAE < 1",
            "observed": f"{p2_ratio:.3f}",
            "passed": bool(p2_ratio < 1.0),
        }
    )

    p3_parts = []
    p3_passed = True
    for entry in summary["by_share"]:
        if entry["share"] < 0.05:
            continue
        p3_parts.append(f"{entry['share']:.2f}: {entry['added']:+.2%}")
        if not entry["added"] < 0.05:
            p3_passed = False
    verdicts.append(
        {
            "id": "P3",
            "claim": "forecast error added by the partial view < 5% at every share >= 0.05",
            "observed": "; ".join(p3_parts),
            "passed": p3_passed,
        }
    )

    naive_coverage = store["naive"]["cov80"]
    achievable = store["naive"]["achievable80"]
    verdicts.append(
        {
            "id": "P4",
            "claim": "seasonal naive 80% coverage <= achievable 80% coverage - 0.10",
            "observed": f"{naive_coverage:.3f} vs achievable {achievable:.3f}",
            "passed": bool(naive_coverage <= achievable - 0.10),
        }
    )

    p5_parts = []
    p5_passed = True
    p5_series = ("store_est", "vendor")
    for series in p5_series:
        oracle = tables[series]["oracle"]
        level_names = ("80", "95")
        for level_name in level_names:
            coverage = oracle["cov" + level_name]
            expected = oracle["achievable" + level_name]
            standard_error = oracle["se" + level_name]
            z_score = (coverage - expected) / standard_error
            p5_parts.append(
                f"{series} {level_name}%: {coverage:.3f} vs {expected:.3f} (z {z_score:+.2f})"
            )
            if abs(z_score) > 3.0:
                p5_passed = False
    verdicts.append(
        {
            "id": "P5",
            "claim": "oracle 80%/95% coverage within 3 Poisson-binomial SE of achievable",
            "observed": "; ".join(p5_parts),
            "passed": p5_passed,
        }
    )

    b1_passed = True
    b1_parts = []
    for series in SERIES:
        oracle_mae = tables[series]["oracle"]["mae"]
        best_other = math.inf
        for model in MODELS:
            if model == "oracle":
                continue
            best_other = min(best_other, tables[series][model]["mae"])
        b1_parts.append(f"{series} {oracle_mae:.2f} vs best other {best_other:.2f}")
        if not oracle_mae < best_other:
            b1_passed = False
    verdicts.append(
        {
            "id": "B1",
            "claim": "the oracle has the lowest pooled MAE in every series",
            "observed": "; ".join(b1_parts),
            "passed": b1_passed,
        }
    )
    return verdicts


def verdict_word(passed):
    if passed:
        return "PASS"
    return "FAIL"


def verdict_lines(summary):
    lines = []
    for verdict in summary["predictions"]:
        lines.append(
            f"{verdict['id']}: {verdict['claim']}. Observed: {verdict['observed']} -> "
            f"{verdict_word(verdict['passed'])}"
        )
    return lines


def readme_headline_lines(summary):
    """The two lines at the top of the README (engineer's, then owner's)."""
    headline = summary["headline"]
    owner = summary["owner"]
    setup = summary["setup"]
    first = (
        f"**Forecast error added by the partial view: {headline['added']:+.2%}** "
        f"{null_clause(headline)}. A weekday-average forecast of daily store orders trained on the history this "
        f"repo estimates from one vendor's own sales (vendor share {setup['main_share']:.2f}) "
        f"misses by {headline['mae_est']:.2f} orders a day (MAE), vs "
        f"{headline['mae_true']:.2f} when trained on the true history "
        f"(simulated store, {setup['n_test_days']} test days, Results section 4)."
    )
    second = (
        f"Next week's orders, day by day: the daily forecasts miss "
        f"{owner['store_wape_weekday']:.2%} vs {owner['store_wape_naive']:.2%} for last "
        f"week's numbers (WAPE over the 7 days, store total; not the error on the week's "
        f"sum); your own orders: {owner['vendor_wape_weekday']:.2%} vs "
        f"{owner['vendor_wape_naive']:.2%}."
    )
    return [first, second]


def why_lines(key):
    why = WHY_THIS_METRIC[key]
    return [
        f"> **Why this metric.** Owner's question: {why['question']}",
        f"> Kind of number: {why['kind']}",
        f"> What would make it lie: {why['lie']}",
    ]


def model_row(label, stats, is_naive):
    if is_naive:
        beats = "baseline (1.000)"
    elif stats["ratio_vs_naive"] < 1.0:
        beats = f"yes ({stats['ratio_vs_naive']:.3f})"
    else:
        beats = f"no ({stats['ratio_vs_naive']:.3f})"
    return (
        f"| {label} | {stats['mae_mean']:.2f} ± {stats['mae_sd']:.2f} | {stats['wape']:.2%} "
        f"| {stats['mase']:.3f} | {beats} | {stats['cov80']:.3f} | {stats['cov95']:.3f} "
        f"| {stats['achievable80']:.3f} / {stats['achievable95']:.3f} |"
    )


def kpi_lines(summary):
    """Shared-standard (a) KPI table for the forecasting KPIs."""
    headline = summary["headline"]
    owner = summary["owner"]
    store = summary["tables"]["store_est"]
    lines = [
        (
            "| KPI | formula | why it matters | compared against | what the software does "
            "to it | measured on | value now |"
        ),
        "|---|---|---|---|---|---|---|",
        (
            "| **Forecast error added by the partial view** (headline) | weekday-average MAE "
            "trained on re-estimated history ÷ the same trained on true history − 1 "
            "(same training and test days) | near 0 means one vendor can forecast the whole "
            "store from its own receipts | the same forecaster trained on full truth "
            "| rebuilds the history it learns from | synthetic truth "
            f"| {headline['added']:+.2%} (results/forecast_summary.json: headline.added) |"
        ),
        (
            "| Next-week orders, owner view | WAPE = Σ\\|daily error\\| ÷ Σ daily actual over "
            "the next 7 days' store orders (day by day, not the error on the week's sum) "
            "| \"% of orders missed\"; defined on zero days | seasonal naive "
            "| fewer orders missed | synthetic truth "
            f"| {owner['store_wape_weekday']:.2%} vs {owner['store_wape_naive']:.2%} "
            "(results/forecast_summary.json: owner) |"
        ),
        (
            "| Beats naive out of sample? | test MAE ÷ seasonal-naive test MAE; MASE beside it "
            "| does it beat what anyone would do by hand? | seasonal naive, same test days "
            "| below 1 means the software wins | synthetic truth "
            f"| {store['weekday_avg']['ratio_vs_naive']:.3f}; MASE "
            f"{store['weekday_avg']['mase']:.3f} "
            "(results/forecast_summary.json: tables.store_est.weekday_avg) |"
        ),
        (
            "| 80% interval coverage vs achievable | share of test days inside the 80% "
            "interval, beside the oracle's expected coverage | restock and cash need a range "
            "| the oracle's achievable coverage | shows whether the interval is honest "
            f"| synthetic truth | {store['weekday_avg']['cov80']:.3f} vs "
            f"{store['weekday_avg']['achievable80']:.3f} "
            "(results/forecast_summary.json: tables.store_est.weekday_avg) |"
        ),
    ]
    return lines


def forecast_markdown(summary):
    """README Results section 4, exactly as the README copies it."""
    setup = summary["setup"]
    headline = summary["headline"]
    owner = summary["owner"]
    diagnostics = summary["diagnostics"]
    seeds = setup["seeds"]
    lines = []
    lines.append("### 4. Forecasting daily orders (rolling-origin backtest)")
    lines.append("")
    lines.append(
        f"Setup: the frozen config above at vendor share {setup['main_share']:.2f}; "
        f"forecasting seeds {seeds[0]}..{seeds[-1]} (pre-registered in PREDICTIONS.md); "
        f"{setup['warmup_days']}-day warm-up, {setup['horizon_days']}-day horizon, an origin "
        f"every {setup['step_days']} days: {setup['n_origins_per_seed']} origins per seed, "
        f"{setup['n_folds']} folds, {setup['n_test_days']} test days per series. At every "
        "origin the estimator is re-run on the orders seen before it, and the incomplete "
        "last day is dropped from both store histories. Regenerate with "
        "`python -m ordergaps.evaluation.forecast_experiment --outdir results` (writes "
        "results/, which this section copies; `--check` verifies the committed copy)."
    )
    lines.append("")
    lines.append(
        f"Headline: forecast error added by the partial view = {headline['added']:+.2%} "
        f"(weekday average: MAE {headline['mae_est']:.2f} trained on the estimated history "
        f"vs {headline['mae_true']:.2f} trained on the true history), "
        f"{null_clause(headline)}."
    )
    lines.append("")
    lines.append(
        f"Owner view: next week's store orders, day by day, the weekday average's daily "
        f"forecasts miss {owner['store_wape_weekday']:.2%} (WAPE) vs {owner['store_wape_naive']:.2%} for "
        f"last week's number; your own orders, {owner['vendor_wape_weekday']:.2%} vs "
        f"{owner['vendor_wape_naive']:.2%}."
    )
    lines.append("")
    lines.append("#### 4.0 Forecast KPIs")
    lines.append("")
    kpi_table = kpi_lines(summary)
    lines.extend(kpi_table)
    lines.append("")
    lines.append(
        "Guarantees (tests, not KPIs): no look-ahead (changing orders after an origin leaves "
        "that fold unchanged; a run-once implementation fails the same test); full-view "
        "identity (share 1.0, no voids: the headline is exactly 0); poisson_quantile(10, "
        "0.025) = 4 and poisson_quantile(10, 0.975) = 17; seasonal naive has zero error on "
        "an exactly weekly series."
    )
    lines.append("")
    for series in SERIES:
        lines.append(f"#### {SERIES_TITLES[series]}")
        lines.append("")
        lines.append(
            "| model | MAE mean ± sd | WAPE | MASE | beats naive? (MAE ratio) "
            "| 80% coverage | 95% coverage | achievable 80% / 95% |"
        )
        lines.append("|---|---|---|---|---|---|---|---|")
        series_table = summary["tables"][series]
        for model in MODELS:
            row = model_row(MODEL_LABELS[model], series_table[model], model == "naive")
            lines.append(row)
        lines.append("")
        why = why_lines(series)
        lines.extend(why)
        lines.append("")
    lines.append("#### 4.4 Forecast error added by the partial view, by share")
    lines.append("")
    lines.append(
        "| vendor share | MAE trained on estimates | MAE trained on truth | error added |"
    )
    lines.append("|---|---|---|---|")
    for entry in summary["by_share"]:
        lines.append(
            f"| {entry['share']:.2f} | {entry['mae_est']:.2f} | {entry['mae_true']:.2f} "
            f"| {entry['added']:+.2%} |"
        )
    lines.append("")
    if headline["is_null"]:
        straddle_text = "straddle"
    else:
        straddle_text = "do not straddle"
    lines.append(
        f"Spread at vendor share {setup['main_share']:.2f}: the same ratio pooled per seed "
        f"runs between {headline['seed_added_min']:+.2%} and "
        f"{headline['seed_added_max']:+.2%} across the {headline['n_seeds']} seeds (sd "
        f"{headline['seed_added_sd']:.2%}), so the headline {headline['added']:+.2%} is "
        f"{null_word(headline)}: the per-seed values "
        f"{straddle_text} 0."
    )
    lines.append("")
    lines.extend(why_lines("by_share"))
    lines.append("")
    lines.append(f"#### 4.5 Backtest checks (vendor share {setup['main_share']:.2f})")
    lines.append("")
    lines.append("| check | value |")
    lines.append("|---|---|")
    lines.append(
        f"| training days a single run over all the data would rewrite "
        f"| {diagnostics['leak_days_changed']} of {diagnostics['leak_days_compared']} |"
    )
    lines.append(
        f"| learned weekday paces it would rewrite "
        f"| {diagnostics['leak_rates_changed']} of {diagnostics['leak_rates_compared']} |"
    )
    lines.append(
        f"| dropped last day: estimated ÷ true orders | {diagnostics['dropped_ratio']:.4f} |"
    )
    lines.append(
        f"| training days per fold (mean); share of them full days "
        f"| {diagnostics['train_days_mean']:.2f}; {diagnostics['full_train_fraction']:.2%} |"
    )
    lines.append(
        f"| test days with zero actual orders (all three series) "
        f"| {diagnostics['zero_actual_test_days']} |"
    )
    lines.append("")
    lines.extend(why_lines("diagnostics"))
    lines.append("")
    lines.append("#### 4.6 Pre-registered predictions (PREDICTIONS.md, 2026-11-04)")
    lines.append("")
    verdicts = verdict_lines(summary)
    for verdict in verdicts:
        lines.append(f"- {verdict}")
    lines.append("")
    return "\n".join(lines)


def build_outputs(shares=SHARES, seeds=FORECAST_SEEDS):
    """{file name: text} for every committed forecast output."""
    folds, origins = run_forecast_experiment(shares, seeds)
    folds_text = frame_to_csv_text(folds)
    origins_text = frame_to_csv_text(origins)
    outputs = outputs_from_csv(folds_text, origins_text)
    return outputs


def outputs_from_csv(folds_text, origins_text):
    """Summary, JSON and markdown from the CSV texts alone (the committed chain)."""
    folds = csv_text_to_frame(folds_text)
    origins = csv_text_to_frame(origins_text)
    summary = summarize(folds, origins)
    rounded = round_structure(summary)
    rounded["readme_headline"] = readme_headline_lines(summary)
    summary_text = json.dumps(rounded, indent=2, sort_keys=True) + "\n"
    markdown_text = forecast_markdown(summary)
    return {
        FOLDS_FILE: folds_text,
        ORIGINS_FILE: origins_text,
        SUMMARY_FILE: summary_text,
        MARKDOWN_FILE: markdown_text,
    }


def write_outputs(outputs, outdir):
    os.makedirs(outdir, exist_ok=True)
    names = sorted(outputs)
    for name in names:
        path = os.path.join(outdir, name)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(outputs[name])


def _numbers_close(left, right):
    if isinstance(left, str) or isinstance(right, str):
        return left == right
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    left_number = float(left)
    right_number = float(right)
    if math.isnan(left_number) and math.isnan(right_number):
        return True
    return math.isclose(left_number, right_number, rel_tol=1e-9, abs_tol=1e-12)


def structures_close(left, right):
    """Equal up to 1e-9 relative on numbers (libm ulps across platforms)."""
    if isinstance(left, dict) and isinstance(right, dict):
        if sorted(left) != sorted(right):
            return False
        keys = sorted(left)
        for key in keys:
            if not structures_close(left[key], right[key]):
                return False
        return True
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        pairs = zip(left, right)
        for left_item, right_item in pairs:
            if not structures_close(left_item, right_item):
                return False
        return True
    return _numbers_close(left, right)


def frames_close(left_text, right_text):
    left = csv_text_to_frame(left_text)
    right = csv_text_to_frame(right_text)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    left_records = left.to_dict(orient="records")
    right_records = right.to_dict(orient="records")
    return structures_close(left_records, right_records)


def stale_outputs(fresh, outdir):
    """Names of committed files that differ from a fresh run (empty = fresh).

    Markdown must match byte for byte (it is what the README copies); CSV
    and JSON numbers must match to 1e-9 relative.
    """
    stale = []
    names = sorted(fresh)
    for name in names:
        path = os.path.join(outdir, name)
        if not os.path.isfile(path):
            stale.append(name)
            continue
        with open(path, encoding="utf-8") as handle:
            committed = handle.read()
        if name == MARKDOWN_FILE:
            matches = committed == fresh[name]
        elif name.endswith(".csv"):
            matches = frames_close(committed, fresh[name])
        else:
            matches = structures_close(json.loads(committed), json.loads(fresh[name]))
        if not matches:
            stale.append(name)
    return stale


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed outputs are stale"
    )
    args = parser.parse_args(argv)
    start_time = time.monotonic()
    outputs = build_outputs()
    elapsed_seconds = time.monotonic() - start_time
    if args.check:
        stale = stale_outputs(outputs, args.outdir)
        if stale:
            print(f"stale forecast outputs in {args.outdir}: {', '.join(stale)}")
            return 1
        print(f"forecast outputs in {args.outdir} are fresh ({elapsed_seconds:.1f} s)")
        return 0
    write_outputs(outputs, args.outdir)
    print(outputs[MARKDOWN_FILE])
    print(f"wrote {args.outdir}/ in {elapsed_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
