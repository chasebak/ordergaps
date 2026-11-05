"""The Gap Explorer page data: python -m ordergaps.viz build [--check].

Writes, from the simulator and the estimator only (the build takes no input
file and reads nothing from disk):
  docs/data/summary.json     per vendor share: the README's section 1-3
                             numbers (error curve, uniform baseline, full-day
                             yield, length bias, counted-vs-allocated rows as
                             the README prints them), the acceptance band,
                             the why-box text, provenance labels, footer
  docs/data/share_grid.json  every frozen run's score (6 shares x 8 seeds),
                             exactly the rows run_matrix() gives the README
  docs/data/week.json        seed 101, ISO 2025-W12 at every share: each
                             day's seen / allocated / true orders and label,
                             the learned vs true open-hours pace, and the
                             largest gap that starts each day with its
                             counted size, simulator count and modelled split
  docs/data/meta.json        the pinned tokens.css hash, the Chart.js tag and
                             the measured data sizes
JSON: sorted keys, floats rounded to 10 significant digits, NaN as null.

Why one week of one seed: ISO 2025-W12 is the third simulated week, so it
has no edge day (first/last observation) and lies before the planted
numbering reset (week 5 of the frozen config); test_viz.py asserts both.
For a fixed seed the vendor's orders are nested across shares (one shared
uniform per order, simulator.py), so the slider shows the same week
becoming clearer, not six different weeks.

The page (docs/index.html + docs/app.js) computes nothing: it looks values
up in these files and formats them. So the tests are in Python
(tests/test_viz.py): the bundle equals the scoring functions' output at the
same rounding, every day conserves seen + allocated = estimated total, and
every listed gap has unseen = n2 - n1 - 1 = the simulator's count.

--check rebuilds in memory (the frozen matrix, about 10 s, so it is a local
step; CI rebuilds from the session's frozen_records fixture instead) and
exits 1 if a data file is stale or missing, docs/tokens.css differs from the
pinned kit hash, docs/.nojekyll is missing, the data exceed the 300 KB
budget, or a file breaks its key allowlist. Failure mode: --check proves the
JSON is fresh, not that the page reads it correctly; the screenshot
click-through (each slider stop against the JSON) covers that, outside CI.
"""

import argparse
import base64
import hashlib
import json
import math
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from ordergaps.attribution import attribute_gap
from ordergaps.classify import CLOSED
from ordergaps.config import CAVEAT, WEEKDAY_NAMES
from ordergaps.estimator import estimate
from ordergaps.evaluation.experiments import (
    acceptance_check,
    baseline_and_yield,
    counted_vs_allocated,
    error_curve,
    length_bias_summary,
    run_matrix,
    score,
)
from ordergaps.evaluation.run_all import (
    FROZEN_SEEDS,
    SHARES,
    TABLE_SHARE,
    WHY_THIS_METRIC,
    counted_rows,
    frozen_base_config,
    signed_points,
)
from ordergaps.simulator import simulate, with_share

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_PATH = "docs/data/summary.json"
GRID_PATH = "docs/data/share_grid.json"
WEEK_PATH = "docs/data/week.json"
META_PATH = "docs/data/meta.json"
DATA_PATHS = (SUMMARY_PATH, GRID_PATH, WEEK_PATH, META_PATH)
TOKENS_PATH = "docs/tokens.css"
NOJEKYLL_PATH = "docs/.nojekyll"
INDEX_PATH = "docs/index.html"
APP_PATH = "docs/app.js"
GENERATED_BY = "python -m ordergaps.viz build"
# Plan, Shared standards (d): ordergaps' page data stay under 300 KB.
DATA_BUDGET_BYTES = 300_000
SIGNIFICANT_DIGITS = 10
# sha256 of the portfolio page kit's tokens.css, base64 like an SRI hash (a
# 64-hex digest would trip the public privacy test's query-hash pattern).
# docs/tokens.css must be a byte-for-byte copy, so the five pages share one look.
TOKENS_SHA256_B64 = "J3texCt1out7QFMk3jA+rC1vsZ5U8WlE3PnDZFY8OO8="
CHART_JS_VERSION = "4.4.1"
CHART_JS_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
    'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7k'
    'HXO7q3UyZAWw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)
FOOTER = (
    "Every number was computed by this repo's tested Python and copied; the page computes nothing."
)
TITLE = "Gap Explorer"
WEEK_SEED = 101
WEEK_ISO_YEAR = 2025
WEEK_ISO_WEEK = 12
DEFAULT_SHARE = TABLE_SHARE
# A page bundle holds summaries, never order-level rows: no list may be this
# long (the smallest frozen run sees a few hundred orders).
MAX_LIST_LENGTH = 64
PAGE_WHY_KEYS = (
    "weekly_error",
    "uniform_baseline",
    "full_day_yield",
    "week_view",
    "length_bias",
    "worked_gap",
    "counted_vs_allocated",
)
# Key names that would mean basket contents or money left the simulator.
BANNED_KEY_PARTS = ("item", "revenue", "price", "basket", "dollar")
BANNED_KEYS = ("total", "totals")

SUMMARY_KEY_NAMES = (
    "acceptance",
    "by_share",
    "caveat",
    "channel_err_median",
    "channel_err_p25",
    "channel_err_p75",
    "counted",
    "counted_rows",
    "default_share",
    "error",
    "footer",
    "full_day_err_median",
    "full_day_share_mean_abs_err",
    "full_day_yield",
    "gaps",
    "generated_by",
    "hero",
    "inversions",
    "iso_week",
    "iso_year",
    "kind",
    "lie",
    "mean_bias",
    "mean_fraction_observed",
    "median_weekly_err",
    "model_minus_uniform",
    "n_clean_weeks",
    "n_full_days",
    "n_open_days",
    "n_weeks_allocators_differ",
    "n_windows",
    "n_windows_exact",
    "nonfull_day_err_median",
    "nonfull_day_err_p25",
    "nonfull_day_err_p75",
    "p25_bias",
    "p75_bias",
    "pace",
    "passed",
    "provenance",
    "quantity",
    "question",
    "readme_try_it",
    "seed",
    "seeds",
    "share",
    "share_label",
    "shares",
    "slider",
    "spearman",
    "table",
    "table_share",
    "text",
    "tiles",
    "title",
    "uniform_err",
    "uniform_weekly_err",
    "verdict",
    "versus_uniform",
    "week",
    "weekly_err",
    "why",
    "windows",
    "yield_text",
)
SUMMARY_KEYS = set(SUMMARY_KEY_NAMES) | set(PAGE_WHY_KEYS)
GRID_KEYS = {
    "cells",
    "fields",
    "generated_by",
    "seeds",
    "shares",
    "share",
    "seed",
    "fraction_observed",
    "n_weeks_used",
    "weekly_err_median",
    "channel_err_median",
    "rate_bias",
    "n_windows",
    "n_windows_exact",
    "full_day_share_abs_err",
    "full_day_err_median",
    "nonfull_day_err_median",
    "uniform_weekly_err_median",
    "n_weeks_allocators_differ",
    "n_full_days",
    "n_open_days",
}
WEEK_KEYS = {
    "by_share",
    "days",
    "generated_by",
    "iso_week",
    "iso_year",
    "seed",
    "week_days",
    "share",
    "fraction_observed",
    "n_observed",
    "fully_covered",
    "has_anomaly",
    "week_est_total",
    "week_true_total",
    "week_abs_rel_err",
    "n_gaps_touching_week",
    "day",
    "weekday",
    "label",
    "day_text",
    "covered",
    "seen_register",
    "seen_online",
    "unseen_in_store",
    "unseen_online",
    "est_in_store",
    "est_online",
    "est_total",
    "true_in_store",
    "true_online",
    "true_total",
    "seen",
    "allocated",
    "pace",
    "learned_per_hour",
    "true_per_hour",
    "source",
    "open",
    "gaps",
    "n1",
    "n2",
    "t1",
    "t2",
    "unseen",
    "truth_between",
    "kind",
    "split",
    "in_store",
    "online",
    "start_day",
    "score",
    "weekly_err_median",
    "n_windows",
    "n_windows_exact",
    "rate_bias",
}
META_KEYS = {
    "chartjs_tag",
    "chartjs_version",
    "data_budget_bytes",
    "data_bytes",
    "generated_by",
    "tokens_css_sha256_base64",
    SUMMARY_PATH.rsplit("/", 1)[1],
    GRID_PATH.rsplit("/", 1)[1],
    WEEK_PATH.rsplit("/", 1)[1],
}
ALLOWED_KEYS = {
    SUMMARY_PATH: SUMMARY_KEYS,
    GRID_PATH: GRID_KEYS,
    WEEK_PATH: WEEK_KEYS,
    META_PATH: META_KEYS,
}
GRID_FIELDS = (
    "share",
    "seed",
    "fraction_observed",
    "n_weeks_used",
    "weekly_err_median",
    "channel_err_median",
    "rate_bias",
    "n_windows",
    "n_windows_exact",
    "full_day_share_abs_err",
    "full_day_err_median",
    "nonfull_day_err_median",
    "uniform_weekly_err_median",
    "n_weeks_allocators_differ",
    "n_full_days",
    "n_open_days",
)


def page_value(value):
    """value made JSON-ready: floats to 10 significant digits, NaN -> None.

    Rule: ints and bools are kept exactly; floats are rounded with
    format(x, '.10g') so a test can round the function output the same way
    and compare exactly. Failure mode: rounding the JSON one way and the
    test another would let a real difference hide under the tolerance; both
    go through this function.
    """
    if isinstance(value, dict):
        converted = {}
        items = value.items()
        for key, item in items:
            converted[str(key)] = page_value(item)
        return converted
    if isinstance(value, (list, tuple)):
        converted_list = []
        for item in value:
            converted_list.append(page_value(item))
        return converted_list
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number) or math.isinf(number):
            return None
        if number == 0.0:
            return 0.0
        return float(f"{number:.{SIGNIFICANT_DIGITS}g}")
    return value


def json_text(value):
    return json.dumps(page_value(value), indent=1, sort_keys=True, allow_nan=False) + "\n"


def share_label(share):
    """'2%', '10%', ... the slider tick text for a vendor share."""
    return f"{share:.0%}"


def readme_try_it_line():
    """The README's 'Try it' line (tests/test_readme_drift.py requires it verbatim).

    It names the page's folder and how to serve it; it claims no live URL,
    because GitHub Pages is switched on by hand in the repo settings.
    """
    first = share_label(SHARES[0])
    last = share_label(SHARES[-1])
    return (
        f"**Try it (60 s):** the {TITLE} page in `docs/` "
        "(`python -m http.server -d docs`, then open the address it prints) lets you drag "
        f"the share of store orders you see between {first} and {last} and watch one simulated "
        "week become countable; every number on it comes from `python -m ordergaps.viz build`."
    )


def provenance_labels():
    """Per-widget provenance pills, built from the frozen constants."""
    n_shares = len(SHARES)
    n_seeds = len(FROZEN_SEEDS)
    week_text = f"seed {WEEK_SEED} · ISO {WEEK_ISO_YEAR}-W{WEEK_ISO_WEEK:02d}"
    labels = {
        "slider": f"precomputed at {n_shares} shares; stops show as ticks",
        "tiles": f"synthetic truth · {n_seeds} seeds at this share",
        "hero": f"synthetic truth · {n_shares} shares x {n_seeds} seeds",
        "week": f"synthetic truth · {week_text} · one simulated week",
        "pace": f"synthetic truth · {week_text} · learned over all simulated weeks",
        "gaps": f"{week_text} · the largest gap that starts each day",
        "table": f"synthetic truth · {n_seeds} seeds at this share",
    }
    return labels


def share_rows(records):
    """Per vendor share: every estimation number the README prints for it.

    Built only from the README's own functions (error_curve,
    baseline_and_yield, length_bias_summary, counted_vs_allocated and
    run_all.counted_rows) on the same records, so the page cannot disagree
    with README sections 1-3. Display text uses the README's formats.
    Failure mode: a share missing from one of the frames would silently
    shift rows; each lookup is by share value and asserts one match.
    """
    curve = error_curve(records)
    baseline = baseline_and_yield(records)
    bias = length_bias_summary(records)
    rows = []
    for share in SHARES:
        curve_row = _one_row(curve, share)
        baseline_row = _one_row(baseline, share)
        bias_row = _one_row(bias, share)
        table = counted_vs_allocated(records, share)
        section_rows = counted_rows(table)
        table_rows = []
        for quantity, kind, error_text in section_rows:
            table_rows.append({"quantity": quantity, "kind": kind, "error": error_text})
        difference_text = signed_points(baseline_row["model_minus_uniform"])
        text = {
            "windows": f"{table['n_windows_exact']:,} / {table['n_windows']:,}",
            "weekly_err": f"{curve_row['median_weekly_err']:.4%}",
            "uniform_err": f"{baseline_row['uniform_weekly_err']:.4%}",
            "versus_uniform": f"{difference_text} pp: {baseline_row['verdict']}",
            "yield_text": (
                f"{baseline_row['n_full_days']} / {baseline_row['n_open_days']} = "
                f"{baseline_row['full_day_yield']:.1%}"
            ),
            "mean_bias": f"{bias_row['mean_bias']:+.2%}",
        }
        rows.append(
            {
                "share": float(share),
                "share_label": share_label(share),
                "mean_fraction_observed": curve_row["mean_fraction_observed"],
                "median_weekly_err": curve_row["median_weekly_err"],
                "uniform_weekly_err": baseline_row["uniform_weekly_err"],
                "model_minus_uniform": baseline_row["model_minus_uniform"],
                "verdict": baseline_row["verdict"],
                "n_clean_weeks": int(baseline_row["n_clean_weeks"]),
                "n_weeks_allocators_differ": int(baseline_row["n_weeks_allocators_differ"]),
                "n_full_days": int(baseline_row["n_full_days"]),
                "n_open_days": int(baseline_row["n_open_days"]),
                "full_day_yield": baseline_row["full_day_yield"],
                "mean_bias": bias_row["mean_bias"],
                "p25_bias": bias_row["p25_bias"],
                "p75_bias": bias_row["p75_bias"],
                "counted": table,
                "counted_rows": table_rows,
                "text": text,
            }
        )
    return rows


def _one_row(frame, share):
    matches = frame[np.isclose(frame["share"].to_numpy(dtype=float), share)]
    if len(matches) != 1:
        raise ValueError(f"expected one row for share {share}, found {len(matches)}")
    return matches.iloc[0].to_dict()


def summary_bundle(records):
    curve = error_curve(records)
    check = acceptance_check(curve)
    why = {}
    for key in PAGE_WHY_KEYS:
        why[key] = WHY_THIS_METRIC[key]
    share_values = []
    for share in SHARES:
        share_values.append(float(share))
    summary = {
        "title": TITLE,
        "footer": FOOTER,
        "caveat": CAVEAT,
        "generated_by": GENERATED_BY,
        "readme_try_it": readme_try_it_line(),
        "seeds": list(FROZEN_SEEDS),
        "shares": share_values,
        "default_share": float(DEFAULT_SHARE),
        "table_share": float(TABLE_SHARE),
        "week": {"seed": WEEK_SEED, "iso_year": WEEK_ISO_YEAR, "iso_week": WEEK_ISO_WEEK},
        "acceptance": {
            "spearman": check["spearman"],
            "inversions": int(check["inversions"]),
            "passed": bool(check["passed"]),
        },
        "by_share": share_rows(records),
        "why": why,
        "provenance": provenance_labels(),
    }
    return summary


def grid_bundle(records):
    """share_grid.json: run_matrix()'s rows, one cell per (share, seed)."""
    cells = []
    record_rows = records.to_dict("records")
    for record in record_rows:
        cell = {}
        for field in GRID_FIELDS:
            cell[field] = record[field]
        cells.append(cell)
    share_values = []
    for share in SHARES:
        share_values.append(float(share))
    return {
        "generated_by": GENERATED_BY,
        "fields": list(GRID_FIELDS),
        "shares": share_values,
        "seeds": list(FROZEN_SEEDS),
        "cells": cells,
    }


def week_days():
    """The seven dates of ISO WEEK_ISO_YEAR-W{WEEK_ISO_WEEK}, Monday first."""
    monday = date.fromisocalendar(WEEK_ISO_YEAR, WEEK_ISO_WEEK, 1)
    days = []
    offsets = range(7)
    for offset in offsets:
        days.append(monday + timedelta(days=offset))
    return days


def minute_text(timestamp):
    """'2025-03-18 11:04' (minute precision; the page shows it as is)."""
    return timestamp.strftime("%Y-%m-%d %H:%M")


def truth_between(sorted_created_ns, gap):
    """Orders the simulator created strictly between the gap's two seen orders.

    Same rule as experiments.score's window check (voids included: they
    consume numbers). Failure mode: an inclusive bound would count a seen
    order itself and miss by one; score() and this share the bound sides.
    """
    t1 = gap.t1.to_datetime64()
    t2 = gap.t2.to_datetime64()
    count_before_t2 = int(np.searchsorted(sorted_created_ns, t2, side="left"))
    count_through_t1 = int(np.searchsorted(sorted_created_ns, t1, side="right"))
    return count_before_t2 - count_through_t1


def gap_record(gap, result, sorted_created):
    """One worked gap: counted size, simulator count and the modelled split."""
    attribution = attribute_gap(gap, result.sessions, result.rates)
    split_days = set(attribution.in_store_by_day) | set(attribution.online_by_day)
    split = []
    ordered_days = sorted(split_days)
    for day in ordered_days:
        split.append(
            {
                "day": day.isoformat(),
                "in_store": attribution.in_store_by_day.get(day, 0.0),
                "online": attribution.online_by_day.get(day, 0.0),
            }
        )
    return {
        "n1": int(gap.n1),
        "n2": int(gap.n2),
        "t1": minute_text(gap.t1),
        "t2": minute_text(gap.t2),
        "start_day": gap.t1.date().isoformat(),
        "unseen": int(gap.unseen),
        "truth_between": truth_between(sorted_created, gap),
        "kind": attribution.kind,
        "split": split,
    }


def worked_gaps(result, days, sorted_created):
    """Per day of the week: the gap with the most unseen orders that starts that day.

    Rule: largest gap.unseen among gaps whose t1 falls on the day; ties go
    to the earliest t1. Days where no gap starts are skipped. Why the
    largest: it is the gap with the most orders for the model to place
    (at low shares it usually crosses closed hours, a "mixed" gap), so it
    shows counted vs modelled most plainly. Failure mode: a page listing every gap would grow with
    the share (n_gaps_touching_week in week.json) and break the size budget.
    """
    best_by_day = {}
    for gap in result.gaps:
        day = gap.t1.date()
        if day not in days:
            continue
        current = best_by_day.get(day)
        if current is None:
            best_by_day[day] = gap
            continue
        is_larger = gap.unseen > current.unseen
        is_earlier_tie = gap.unseen == current.unseen and gap.t1 < current.t1
        if is_larger or is_earlier_tie:
            best_by_day[day] = gap
    records = []
    for day in days:
        gap = best_by_day.get(day)
        if gap is None:
            continue
        records.append(gap_record(gap, result, sorted_created))
    return records


def week_view(share):
    """week.json's entry for one share: seed WEEK_SEED's ISO week, scored.

    Returns (entry, metrics) where metrics = experiments.score() on the same
    sim and result, i.e. the share_grid cell for (share, WEEK_SEED).
    """
    config = with_share(frozen_base_config(), share)
    sim = simulate(config, WEEK_SEED)
    result = estimate(sim.observed, config.store_hours)
    metrics = score(config, sim, result)
    days = week_days()
    day_set = set(days)

    truth_by_day = {}
    truth_tuples = list(sim.truth_daily.itertuples(index=False))
    for row in truth_tuples:
        truth_by_day[row.day] = row
    day_entries = []
    daily_tuples = list(result.daily.itertuples(index=False))
    for row in daily_tuples:
        if row.day not in day_set:
            continue
        truth = truth_by_day[row.day]
        seen = int(row.seen_register) + int(row.seen_online)
        allocated = float(row.unseen_in_store) + float(row.unseen_online)
        day_entries.append(
            {
                "day": row.day.isoformat(),
                "weekday": row.weekday,
                "day_text": f"{row.weekday} {row.day.strftime('%m-%d')}",
                "label": row.label,
                "covered": bool(row.covered),
                "seen_register": int(row.seen_register),
                "seen_online": int(row.seen_online),
                "seen": seen,
                "unseen_in_store": float(row.unseen_in_store),
                "unseen_online": float(row.unseen_online),
                "allocated": allocated,
                "est_in_store": float(row.est_in_store),
                "est_online": float(row.est_online),
                "est_total": float(row.est_total),
                "true_in_store": int(truth.in_store),
                "true_online": int(truth.online),
                "true_total": int(truth.total),
            }
        )

    weekly = result.weekly
    week_rows = weekly[
        (weekly["iso_year"] == WEEK_ISO_YEAR) & (weekly["iso_week"] == WEEK_ISO_WEEK)
    ]
    week_row = week_rows.iloc[0]
    truth_weekly = sim.truth_weekly
    truth_rows = truth_weekly[
        (truth_weekly["iso_year"] == WEEK_ISO_YEAR) & (truth_weekly["iso_week"] == WEEK_ISO_WEEK)
    ]
    week_true_total = int(truth_rows.iloc[0]["total"])
    week_est_total = float(week_row["est_total"])

    pace = []
    weekday_values = range(7)
    for weekday in weekday_values:
        scheduled = config.store_hours.for_weekday(weekday)
        if scheduled is None:
            pace.append(
                {
                    "weekday": WEEKDAY_NAMES[weekday],
                    "open": False,
                    "learned_per_hour": None,
                    "true_per_hour": None,
                    "source": CLOSED,
                }
            )
            continue
        learned_per_hour = result.rates.rate_for(weekday) * 60.0
        true_per_hour = sim.true_weekday_rates[weekday] * 60.0
        pace.append(
            {
                "weekday": WEEKDAY_NAMES[weekday],
                "open": True,
                "learned_per_hour": learned_per_hour,
                "true_per_hour": true_per_hour,
                "source": result.rates.source[weekday],
            }
        )

    week_start = days[0]
    week_end = days[-1] + timedelta(days=1)
    n_touching = 0
    for gap in result.gaps:
        if gap.t2.date() >= week_start and gap.t1.date() < week_end:
            n_touching += 1

    sorted_created = np.sort(sim.orders["created_at"].to_numpy())
    entry = {
        "share": float(share),
        "fraction_observed": sim.fraction_observed,
        "n_observed": len(sim.observed),
        "fully_covered": bool(week_row["fully_covered"]),
        "has_anomaly": bool(week_row["has_anomaly"]),
        "week_est_total": week_est_total,
        "week_true_total": week_true_total,
        "week_abs_rel_err": abs(week_est_total - week_true_total) / week_true_total,
        "n_gaps_touching_week": n_touching,
        "days": day_entries,
        "pace": pace,
        "gaps": worked_gaps(result, days, sorted_created),
        "score": {
            "weekly_err_median": metrics.weekly_err_median,
            "n_windows": metrics.n_windows,
            "n_windows_exact": metrics.n_windows_exact,
            "rate_bias": metrics.rate_bias,
        },
    }
    return entry, metrics


def week_bundle():
    entries = []
    for share in SHARES:
        entry, _metrics = week_view(share)
        entries.append(entry)
    day_texts = []
    days = week_days()
    for day in days:
        day_texts.append(day.isoformat())
    return {
        "generated_by": GENERATED_BY,
        "seed": WEEK_SEED,
        "iso_year": WEEK_ISO_YEAR,
        "iso_week": WEEK_ISO_WEEK,
        "week_days": day_texts,
        "by_share": entries,
    }


def week_json_text():
    """week.json's text alone (seconds; the TZ determinism test runs it twice)."""
    return json_text(week_bundle())


def frozen_records():
    """The README's frozen matrix: run_matrix(frozen_base_config(), SHARES, FROZEN_SEEDS)."""
    return run_matrix(frozen_base_config(), SHARES, FROZEN_SEEDS)


def sha256_base64(data):
    digest = hashlib.sha256(data).digest()
    return base64.b64encode(digest).decode("ascii")


def text_bytes(text):
    return len(text.encode("utf-8"))


def build_outputs(records=None):
    """{repo-relative path: text} for every data file. records defaults to
    the frozen matrix (pass tests' frozen_records fixture to skip the ~10 s)."""
    if records is None:
        records = frozen_records()
    outputs = {
        SUMMARY_PATH: json_text(summary_bundle(records)),
        GRID_PATH: json_text(grid_bundle(records)),
        WEEK_PATH: week_json_text(),
    }
    data_bytes = {}
    data_names = (SUMMARY_PATH, GRID_PATH, WEEK_PATH)
    for path in data_names:
        name = path.rsplit("/", 1)[1]
        data_bytes[name] = text_bytes(outputs[path])
    meta = {
        "generated_by": GENERATED_BY,
        "chartjs_version": CHART_JS_VERSION,
        "chartjs_tag": CHART_JS_TAG,
        "tokens_css_sha256_base64": TOKENS_SHA256_B64,
        "data_budget_bytes": DATA_BUDGET_BYTES,
        "data_bytes": data_bytes,
    }
    outputs[META_PATH] = json_text(meta)
    return outputs


def collect_keys(value, found):
    """Every dict key anywhere inside value, added to the set found."""
    if isinstance(value, dict):
        items = value.items()
        for key, item in items:
            found.add(key)
            collect_keys(item, found)
    elif isinstance(value, list):
        for item in value:
            collect_keys(item, found)
    return found


def longest_list(value):
    """Length of the longest list anywhere inside value."""
    longest = 0
    if isinstance(value, dict):
        children = list(value.values())
    elif isinstance(value, list):
        longest = len(value)
        children = value
    else:
        children = []
    for child in children:
        child_longest = longest_list(child)
        longest = max(longest, child_longest)
    return longest


def banned_keys(keys):
    """Keys that name basket contents or money (the bundle exports neither)."""
    found = []
    ordered = sorted(keys)
    for key in ordered:
        lowered = key.lower()
        if lowered in BANNED_KEYS:
            found.append(key)
            continue
        for part in BANNED_KEY_PARTS:
            if part in lowered:
                found.append(key)
                break
    return found


def bundle_problems(outputs):
    """Allowlist, banned-key, list-length and size problems of a built bundle."""
    problems = []
    total_bytes = 0
    paths = sorted(outputs)
    for path in paths:
        text = outputs[path]
        total_bytes += text_bytes(text)
        value = json.loads(text)
        keys = collect_keys(value, set())
        unexpected = sorted(keys - ALLOWED_KEYS[path])
        if unexpected:
            problems.append(f"{path}: keys outside the allowlist: {unexpected}")
        named = banned_keys(keys)
        if named:
            problems.append(f"{path}: basket or money keys: {named}")
        longest = longest_list(value)
        if longest > MAX_LIST_LENGTH:
            problems.append(f"{path}: a list of {longest} entries (max {MAX_LIST_LENGTH})")
    if total_bytes > DATA_BUDGET_BYTES:
        problems.append(f"page data are {total_bytes} bytes (budget {DATA_BUDGET_BYTES})")
    return problems


def static_problems(root):
    """Problems with the page's fixed files (tokens copy, .nojekyll, script tag)."""
    problems = []
    tokens_path = root / TOKENS_PATH
    if not tokens_path.is_file():
        problems.append(f"{TOKENS_PATH} is missing")
    elif sha256_base64(tokens_path.read_bytes()) != TOKENS_SHA256_B64:
        problems.append(f"{TOKENS_PATH} is not the page kit's tokens.css")
    if not (root / NOJEKYLL_PATH).is_file():
        problems.append(f"{NOJEKYLL_PATH} is missing")
    index_path = root / INDEX_PATH
    if not index_path.is_file():
        problems.append(f"{INDEX_PATH} is missing")
    elif CHART_JS_TAG not in index_path.read_text(encoding="utf-8"):
        problems.append(f"{INDEX_PATH} lacks the pinned Chart.js tag")
    if not (root / APP_PATH).is_file():
        problems.append(f"{APP_PATH} is missing")
    return problems


def check(root=REPO_ROOT, outputs=None):
    """List of problems; empty means the committed page data are fresh."""
    if outputs is None:
        outputs = build_outputs()
    problems = bundle_problems(outputs)
    paths = sorted(outputs)
    for path in paths:
        file_path = root / path
        if not file_path.is_file():
            problems.append(f"{path} is missing")
            continue
        committed = file_path.read_text(encoding="utf-8")
        if committed != outputs[path]:
            problems.append(f"{path} is stale (run {GENERATED_BY})")
    problems.extend(static_problems(root))
    return problems


def write(root=REPO_ROOT, outputs=None):
    if outputs is None:
        outputs = build_outputs()
    problems = bundle_problems(outputs)
    if problems:
        return problems
    paths = sorted(outputs)
    for path in paths:
        file_path = root / path
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(outputs[path], encoding="utf-8")
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m ordergaps.viz", description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument(
        "--check", action="store_true", help="rebuild in memory and fail if docs/data is stale"
    )
    args = parser.parse_args(argv)
    if args.check:
        problems = check()
        if problems:
            for problem in problems:
                print(problem)
            return 1
        print("docs/data is fresh; tokens.css, .nojekyll and the Chart.js tag check out")
        return 0
    problems = write()
    if problems:
        for problem in problems:
            print(problem)
        return 1
    print(f"wrote {', '.join(DATA_PATHS)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
