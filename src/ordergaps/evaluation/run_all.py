"""Run the evaluation experiments with frozen seeds and write the figures
and the results tables.

    python -m ordergaps.evaluation.run_all              # writes under out/
    python -m ordergaps.evaluation.run_all --outdir figures   # committed figures

Sections 1-3 (estimation) use FROZEN_SEEDS. Section 4 (forecasting) is the
rolling-origin backtest of evaluation.forecast_experiment on its own frozen
seeds; run_all appends it to the results text, writes its files under
--forecast-outdir and prints a PASS/FAIL line for each pre-registered
prediction (PREDICTIONS.md). The committed copy of section 4 lives in
results/ (python -m ordergaps.evaluation.forecast_experiment --outdir results).

Sections 1.1, 1.2 and 1.3 (the uniform-in-time baseline, the full-day
yield and the estimation KPI table) come from the same frozen matrix; run_all writes them into the results text
right after section 1 and, as baseline_yield.md and baseline_yield.json,
under --baseline-outdir. The committed copy lives in results/
(--baseline-outdir results); the README copies baseline_yield.md verbatim.

Seeds, shares and the simulator regime below were frozen BEFORE the
committed results were generated; see README "Design decisions".
"""

import argparse
import json
import math
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ordergaps.config import CAVEAT
from ordergaps.evaluation.experiments import (
    acceptance_check,
    baseline_and_yield,
    counted_vs_allocated,
    error_curve,
    length_bias_summary,
    run_matrix,
)
from ordergaps.simulator import SimConfig

FROZEN_SEEDS = (101, 202, 303, 404, 505, 606, 707, 808)
SHARES = (0.02, 0.05, 0.10, 0.20, 0.35, 0.60)
TABLE_SHARE = 0.10
FLOOR = 1e-6  # log-scale display floor for near-zero errors
BASELINE_MARKDOWN_FILE = "baseline_yield.md"
BASELINE_SUMMARY_FILE = "baseline_yield.json"
BASELINE_REGENERATE = (
    "python -m ordergaps.evaluation.run_all --outdir figures --baseline-outdir results"
)
VERDICT_ORDER = ("model lower", "tie", "uniform lower")
# The why boxes baseline_yield.json carries (sections 1.1 and 1.2 only).
BASELINE_WHY_KEYS = ("uniform_baseline", "full_day_yield")
# Results section -> its why box, printed under the section's table.
SECTION_WHY_KEYS = {"1": "weekly_error", "2": "length_bias", "3": "counted_vs_allocated"}

# The "why this metric" boxes for the estimation sections: one copy.
# Sections 1, 2 and 3 print their box under the table (results_markdown;
# the README copies the same lines, tests/test_readme_drift.py); 1.1 and 1.2
# go into baseline_yield.md / baseline_yield.json.
WHY_THIS_METRIC = {
    "weekly_error": {
        "question": (
            "If I see only my own share of the store's orders, how far off is my weekly "
            "store total?"
        ),
        "kind": (
            "Simulated: the median over seeds of each run's median |est - true| / true over "
            "clean weeks (fully covered, no numbering reset); truth = the simulator's weekly "
            "totals."
        ),
        "lie": (
            "Dropping hard weeks or picking seeds after the fact; the clean-week rule and the "
            "seeds are frozen, tests/test_acceptance.py holds the band and "
            "tests/test_readme_drift.py the printed numbers."
        ),
    },
    "length_bias": {
        "question": "Is the open-hours pace the estimator learns too high or too low?",
        "kind": (
            "Simulated: minutes-weighted mean of (learned - true) / true over open weekdays; "
            "truth = orders the simulator created inside scheduled hours per open minute."
        ),
        "lie": (
            "One seed or one share would hide the spread; the table prints the IQR across "
            "seeds and the figure every seed. The bias is intrinsic (the vendor's own sale "
            "times oversample busy hours), so it is reported, not tuned away."
        ),
    },
    "counted_vs_allocated": {
        "question": "Which of these numbers are counted, and which are the model's guess?",
        "kind": (
            "Counted rows come from order-number arithmetic and are checked window by window "
            "against the orders the simulator created; allocated rows are the timing model's "
            "output, scored against simulator truth with an IQR across seeds."
        ),
        "lie": (
            "A timestamp-precision slip or a mishandled numbering reset would break exactness; "
            "tests/test_acceptance.py requires every window exact at every share and seed."
        ),
    },
    "uniform_baseline": {
        "question": (
            "Is the timing model worth anything for weekly totals, or would spreading "
            "each gap evenly over the clock do as well?"
        ),
        "kind": (
            "Simulated: median |est - true| / true over clean weeks, the same weeks, gaps "
            "and unseen counts for both allocators; truth = the simulator's weekly totals."
        ),
        "lie": (
            "A baseline fed different counts than the model (both must hand out the same "
            "total, and a week no gap straddles must come out identical: tests/test_baseline.py)."
        ),
    },
    "full_day_yield": {
        "question": "How much of the store must I see before a single day's count means anything?",
        "kind": (
            "Counted: the estimator's own day labels (FULL / non-closed days), pooled over "
            "seeds; no truth involved, so section 3's full-day error is what says FULL is earned."
        ),
        "lie": (
            "A looser pin rule would raise the yield without making full days more exact "
            "(section 3 prints the per-day error on full days, measured against truth)."
        ),
    },
}


def frozen_base_config():
    return SimConfig(
        n_weeks=10,
        vendor_share=0.10,
        void_rate=0.02,
        reset_week=5,
        drift_per_week=0.01,
    )


def hero_figure(records, curve, check, path):
    figure, axes = plt.subplots(figsize=(9.0, 5.0), dpi=100)
    x_points = records["fraction_observed"].clip(lower=FLOOR)
    y_points = records["weekly_err_median"].clip(lower=FLOOR)
    axes.scatter(x_points, y_points, s=24, alpha=0.45, color="#7193b8", label="one seed")
    x_curve = curve["mean_fraction_observed"].clip(lower=FLOOR)
    y_curve = curve["median_weekly_err"].clip(lower=FLOOR)
    axes.plot(
        x_curve,
        y_curve,
        marker="o",
        color="#1f4e79",
        linewidth=2.0,
        label="median across seeds",
    )
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("fraction of store orders observed (the vendor's own sales)")
    axes.set_ylabel("median |relative error| of weekly store volume")
    axes.set_title("Seeing more of the sequence makes the volume estimate better")
    label = f"Spearman rho = {check['spearman']:.2f}, inversions = {check['inversions']}"
    axes.annotate(label, xy=(0.03, 0.06), xycoords="axes fraction", fontsize=10)
    axes.grid(True, which="both", alpha=0.25)
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def length_bias_figure(records, summary, path):
    figure, axes = plt.subplots(figsize=(9.0, 5.0), dpi=100)
    axes.scatter(
        records["share"],
        records["rate_bias"],
        s=24,
        alpha=0.45,
        color="#b87171",
        label="one seed",
    )
    axes.plot(
        summary["share"],
        summary["mean_bias"],
        marker="o",
        color="#792020",
        linewidth=2.0,
        label="mean across seeds",
    )
    axes.axhline(0.0, color="#444444", linewidth=1.0, linestyle="--")
    axes.set_xscale("log")
    axes.set_xlabel("vendor share of store orders")
    axes.set_ylabel("relative bias of learned open-hours pace")
    axes.set_title("Length bias: rates learned from the vendor's own sale times")
    axes.grid(True, which="both", alpha=0.25)
    axes.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def why_lines(key):
    why = WHY_THIS_METRIC[key]
    return [
        f"> **Why this metric.** Owner's question: {why['question']}",
        f"> Kind of number: {why['kind']}",
        f"> What would make it lie: {why['lie']}",
    ]


def verdict_summary_sentence(baseline):
    """'Model lower at vendor share 0.02; tie at ...' in VERDICT_ORDER."""
    shares_by_verdict = {}
    for verdict in VERDICT_ORDER:
        shares_by_verdict[verdict] = []
    baseline_tuples = list(baseline.itertuples(index=False))
    for row in baseline_tuples:
        shares_by_verdict[row.verdict].append(f"{row.share:.2f}")
    parts = []
    for verdict in VERDICT_ORDER:
        shares = shares_by_verdict[verdict]
        if shares:
            parts.append(f"{verdict} at vendor share {', '.join(shares)}")
    sentence = "; ".join(parts)
    return sentence[0].upper() + sentence[1:] + "."


def signed_points(fraction_difference):
    """A difference of two fractions in percentage points, 4 decimals, signed;
    a difference that rounds to zero prints as 0.0000 (no stray -0.0000)."""
    text = f"{fraction_difference * 100.0:+.4f}"
    if text in ("+0.0000", "-0.0000"):
        return "0.0000"
    return text


def share_range_text(baseline, column, number_format):
    """'from X at 0.02 to Y at 0.60': the first and last share's value.

    Written as "from X at a to Y at b", never as two percents joined by an
    arrow: that shape is reserved for store-share changes and banned in
    public text (tests/test_privacy_patterns.py).
    """
    baseline_tuples = list(baseline.itertuples(index=False))
    first = baseline_tuples[0]
    last = baseline_tuples[-1]
    first_value = format(getattr(first, column), number_format)
    last_value = format(getattr(last, column), number_format)
    return f"from {first_value} at {first.share:.2f} to {last_value} at {last.share:.2f}"


def is_sub_percent(values):
    for value in values:
        if not value < 0.01:
            return False
    return True


def is_rising(values):
    previous = None
    for value in values:
        if previous is not None and not value > previous:
            return False
        previous = value
    return True


def estimation_kpi_lines(baseline, table):
    """Shared-standard (a) KPI table for the estimation KPIs (README 1.3).

    Rows: weekly volume error vs share (model and the uniform-in-time
    baseline, section 1.1), full-day yield (section 1.2) and the channel-split
    error (section 3), then the window-exactness guarantee. The "what the
    software does" cells are checked, not asserted: "sub-percent at every
    share" and "rises with share" print only when the numbers say so.
    Failure mode: the value-now cells quote the end shares only; the full
    per-share rows are in sections 1.1-1.2, which the same tests pin.
    """
    model_errors = list(baseline["model_weekly_err"])
    yields = list(baseline["full_day_yield"])
    if is_sub_percent(model_errors):
        weekly_effect = "sub-percent at every share"
    else:
        weekly_effect = "above 1% at some share"
    if is_rising(yields):
        yield_effect = "rises with share"
    else:
        yield_effect = "does not rise at every step"
    model_range = share_range_text(baseline, "model_weekly_err", ".4%")
    uniform_range = share_range_text(baseline, "uniform_weekly_err", ".4%")
    yield_range = share_range_text(baseline, "full_day_yield", ".1%")
    sentence = verdict_summary_sentence(baseline)
    verdicts = sentence[0].lower() + sentence[1:-1]
    seeds_text = f"seeds {FROZEN_SEEDS[0]}..{FROZEN_SEEDS[-1]}"
    lines = [
        (
            "| KPI | formula | why it matters | compared against | what the software does "
            "to it | measured on | value now |"
        ),
        "|---|---|---|---|---|---|---|",
        (
            "| Weekly volume error vs share | median \\|est − true\\| ÷ true over clean weeks "
            "| \"How wrong am I at MY share?\" | truth; the uniform-in-time baseline "
            f"(section 1.1) | {weekly_effect} | synthetic truth ({seeds_text}) "
            f"| model {model_range}; uniform {uniform_range}; {verdicts} "
            "(results/baseline_yield.json: rows[*].model_weekly_err, "
            "rows[*].uniform_weekly_err) |"
        ),
        (
            "| Full-day yield | FULL days ÷ non-closed days, pooled over seeds "
            "| how much share you need before day-level numbers mean anything "
            f"| across shares | {yield_effect} | the estimator's own day labels ({seeds_text}) "
            f"| {yield_range} (results/baseline_yield.json: rows[*].full_day_yield) |"
        ),
        (
            "| Channel-split error | median \\|est in-store − true in-store\\| ÷ true in-store, "
            "weekly | the in-store share is the number used with the store, so its error bar "
            "must hold up | truth | the model's weakest output, shown openly "
            f"| synthetic truth ({seeds_text}) "
            f"| {table['channel_err_median']:.2%} [IQR {table['channel_err_p25']:.2%}, "
            f"{table['channel_err_p75']:.2%}] at vendor share {table['table_share']:.2f} "
            "(results/baseline_yield.json: kpi.channel_err_median, kpi.channel_err_p25, "
            "kpi.channel_err_p75) |"
        ),
    ]
    guarantee = (
        "Guarantees (tests, not KPIs): window exactness, "
        f"{table['n_windows_exact']}/{table['n_windows']} windows between seen orders match "
        f"ground truth at vendor share {table['table_share']:.2f} (results/baseline_yield.json: "
        "kpi.n_windows_exact, kpi.n_windows); tests/test_acceptance.py requires every window "
        "exact at every share and seed. It is exact by construction, so it is a test, not a KPI."
    )
    return lines, guarantee


def baseline_markdown(baseline, table):
    """README sections 1.1, 1.2 and 1.3 (results/baseline_yield.md), from
    experiments.baseline_and_yield() and counted_vs_allocated()."""
    lines = []
    lines.append("#### 1.1 Against a uniform-in-time baseline")
    lines.append("")
    lines.append(
        "The baseline takes the same gaps and the same unseen counts and spreads each gap "
        "evenly over its clock time, ignoring store hours, sessions and learned paces "
        "(`ordergaps.baseline`). Same seeds, same clean weeks. Regenerate with "
        f"`{BASELINE_REGENERATE}`."
    )
    lines.append("")
    lines.append(
        "| vendor share | model (this repo) | uniform-in-time baseline "
        "| vs uniform (model - uniform) | clean weeks whose totals differ |"
    )
    lines.append("|---|---|---|---|---|")
    baseline_tuples = list(baseline.itertuples(index=False))
    for row in baseline_tuples:
        difference_text = signed_points(row.model_minus_uniform)
        lines.append(
            f"| {row.share:.2f} | {row.model_weekly_err:.4%} | {row.uniform_weekly_err:.4%} "
            f"| {difference_text} pp: {row.verdict} "
            f"| {row.n_weeks_allocators_differ} of {row.n_clean_weeks} |"
        )
    lines.append("")
    verdicts = verdict_summary_sentence(baseline)
    lines.append(
        "Totals are arithmetic; the model's value is the split and the day classes. "
        "Both allocators hand out the same unseen orders, so a week's total can differ only "
        f"through a gap that straddles the week's boundary. {verdicts}"
    )
    lines.append("")
    lines.extend(why_lines("uniform_baseline"))
    lines.append("")
    lines.append("#### 1.2 Full-day yield")
    lines.append("")
    lines.append("| vendor share | full days | non-closed days | full-day yield |")
    lines.append("|---|---|---|---|")
    for row in baseline_tuples:
        lines.append(
            f"| {row.share:.2f} | {row.n_full_days} | {row.n_open_days} "
            f"| {row.full_day_yield:.1%} |"
        )
    lines.append("")
    lines.extend(why_lines("full_day_yield"))
    lines.append("")
    lines.append("#### 1.3 Estimation KPIs")
    lines.append("")
    kpi_table, guarantee = estimation_kpi_lines(baseline, table)
    lines.extend(kpi_table)
    lines.append("")
    lines.append(guarantee)
    lines.append("")
    return "\n".join(lines)


def _json_number(value):
    number = float(value)
    if math.isnan(number):
        return None
    return float(f"{number:.10g}")


def baseline_why():
    """The sections 1.1 / 1.2 entries of WHY_THIS_METRIC (baseline_yield.json)."""
    selected = {}
    for key in BASELINE_WHY_KEYS:
        selected[key] = WHY_THIS_METRIC[key]
    return selected


def counted_rows(table):
    """Section 3's rows as (quantity, kind, error text), from counted_vs_allocated().

    The formatter behind the README table (results_markdown). Rule: error
    bars (IQR across seeds) only on the allocated rows; counted rows carry
    exact or sampling-only numbers. Failure mode: a README row typed by hand
    could round differently; tests/test_readme_drift.py compares the README's
    rows to these.
    """
    rows = [
        (
            "window totals between seen orders",
            "counted (ID arithmetic)",
            f"exact: {table['n_windows_exact']}/{table['n_windows']} windows match ground truth",
        ),
        (
            "vendor share of orders, fully-pinned days",
            "counted (sampling noise only)",
            f"mean abs deviation from true share: {table['full_day_share_mean_abs_err']:.4f}",
        ),
        (
            "per-day total volume, full days",
            "counted (boundary spill only)",
            f"median rel err {table['full_day_err_median']:.2%}",
        ),
        (
            "weekly in-store volume (channel split)",
            "allocated (model)",
            (
                f"median rel err {table['channel_err_median']:.2%} "
                f"[IQR {table['channel_err_p25']:.2%}, {table['channel_err_p75']:.2%}]"
            ),
        ),
        (
            "per-day total volume, partial/spread days",
            "allocated (timing model)",
            (
                f"median rel err {table['nonfull_day_err_median']:.2%} "
                f"[IQR {table['nonfull_day_err_p25']:.2%}, {table['nonfull_day_err_p75']:.2%}]"
            ),
        ),
    ]
    return rows


def section_why_block(section):
    """The README why box of results section 1, 2 or 3, as one text block."""
    key = SECTION_WHY_KEYS[section]
    return "\n".join(why_lines(key))


def baseline_summary_json(baseline, table):
    """baseline_yield.json: the rows of sections 1.1 / 1.2, the section-1.3
    KPI values from counted_vs_allocated() ("kpi") and the shared "why this
    metric" text (numbers to 10 significant digits)."""
    rows = []
    baseline_tuples = list(baseline.itertuples(index=False))
    for row in baseline_tuples:
        rows.append(
            {
                "share": _json_number(row.share),
                "model_weekly_err": _json_number(row.model_weekly_err),
                "uniform_weekly_err": _json_number(row.uniform_weekly_err),
                "model_minus_uniform": _json_number(row.model_minus_uniform),
                "verdict": row.verdict,
                "n_clean_weeks": int(row.n_clean_weeks),
                "n_weeks_allocators_differ": int(row.n_weeks_allocators_differ),
                "n_full_days": int(row.n_full_days),
                "n_open_days": int(row.n_open_days),
                "full_day_yield": _json_number(row.full_day_yield),
            }
        )
    summary = {
        "caveat": CAVEAT,
        "seeds": list(FROZEN_SEEDS),
        "regenerate": BASELINE_REGENERATE,
        "rows": rows,
        "kpi": {
            "table_share": _json_number(table["table_share"]),
            "channel_err_median": _json_number(table["channel_err_median"]),
            "channel_err_p25": _json_number(table["channel_err_p25"]),
            "channel_err_p75": _json_number(table["channel_err_p75"]),
            "n_windows_exact": int(table["n_windows_exact"]),
            "n_windows": int(table["n_windows"]),
        },
        "why_this_metric": baseline_why(),
    }
    return json.dumps(summary, indent=2, sort_keys=True) + "\n"


def baseline_outputs(records):
    """{file name: text} for the committed baseline / yield outputs."""
    baseline = baseline_and_yield(records)
    table = counted_vs_allocated(records, TABLE_SHARE)
    return {
        BASELINE_MARKDOWN_FILE: baseline_markdown(baseline, table),
        BASELINE_SUMMARY_FILE: baseline_summary_json(baseline, table),
    }


def results_markdown(curve, check, bias_summary, table, elapsed_seconds, baseline_text=""):
    lines = []
    lines.append("# ordergaps evaluation results")
    lines.append("")
    lines.append(f"Caveat: {CAVEAT}")
    lines.append("")
    lines.append("## 1. Error vs fraction of orders observed")
    lines.append("")
    lines.append("| vendor share | fraction observed | median weekly volume error |")
    lines.append("|---|---|---|")
    curve_tuples = list(curve.itertuples(index=False))
    for row in curve_tuples:
        lines.append(
            f"| {row.share:.2f} | {row.mean_fraction_observed:.3f} "
            f"| {row.median_weekly_err:.4%} |"
        )
    lines.append("")
    lines.append(
        f"Acceptance band: Spearman rho = {check['spearman']:.3f} (required <= -0.8), "
        f"inversions = {check['inversions']} (required <= 1) -> "
        f"{'PASS' if check['passed'] else 'FAIL'}"
    )
    lines.append("")
    lines.append(section_why_block("1"))
    lines.append("")
    if baseline_text:
        lines.append(baseline_text)
    lines.append("## 2. Length bias of learned weekday rates")
    lines.append("")
    lines.append("| vendor share | mean bias | IQR |")
    lines.append("|---|---|---|")
    bias_tuples = list(bias_summary.itertuples(index=False))
    for row in bias_tuples:
        lines.append(
            f"| {row.share:.2f} | {row.mean_bias:+.2%} "
            f"| [{row.p25_bias:+.2%}, {row.p75_bias:+.2%}] |"
        )
    lines.append("")
    lines.append(section_why_block("2"))
    lines.append("")
    lines.append(f"## 3. Counted vs allocated (vendor share {table['table_share']:.2f})")
    lines.append("")
    lines.append("| quantity | kind | error |")
    lines.append("|---|---|---|")
    section_rows = counted_rows(table)
    for quantity, kind, error_text in section_rows:
        lines.append(f"| {quantity} | {kind} | {error_text} |")
    lines.append("")
    lines.append(section_why_block("3"))
    lines.append("")
    lines.append(f"Generated in {elapsed_seconds:.1f} s by `python -m ordergaps.evaluation.run_all`")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.join("out", "figures"))
    parser.add_argument("--report", default=os.path.join("out", "results.md"))
    parser.add_argument("--forecast-outdir", default=os.path.join("out", "forecast"))
    parser.add_argument("--baseline-outdir", default=os.path.join("out", "baseline"))
    args = parser.parse_args(argv)

    start_time = time.monotonic()
    base = frozen_base_config()
    records = run_matrix(base, SHARES, FROZEN_SEEDS)
    curve = error_curve(records)
    check = acceptance_check(curve)
    bias_summary = length_bias_summary(records)
    table = counted_vs_allocated(records, TABLE_SHARE)
    baseline_files = baseline_outputs(records)
    elapsed_seconds = time.monotonic() - start_time

    os.makedirs(args.outdir, exist_ok=True)
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    hero_path = os.path.join(args.outdir, "hero_error_vs_fraction.png")
    bias_path = os.path.join(args.outdir, "length_bias.png")
    hero_figure(records, curve, check, hero_path)
    length_bias_figure(records, bias_summary, bias_path)

    # Imported here, not at the top: forecast_experiment imports this module's
    # frozen config, so a top-level import would be circular.
    from ordergaps.evaluation import forecast_experiment

    forecast_outputs = forecast_experiment.build_outputs()
    forecast_experiment.write_outputs(forecast_outputs, args.forecast_outdir)
    forecast_text = forecast_outputs[forecast_experiment.MARKDOWN_FILE]
    summary_text = forecast_outputs[forecast_experiment.SUMMARY_FILE]
    forecast_summary = json.loads(summary_text)

    os.makedirs(args.baseline_outdir, exist_ok=True)
    baseline_names = sorted(baseline_files)
    for name in baseline_names:
        baseline_path = os.path.join(args.baseline_outdir, name)
        with open(baseline_path, "w", encoding="utf-8") as handle:
            handle.write(baseline_files[name])

    baseline_text = baseline_files[BASELINE_MARKDOWN_FILE]
    text = results_markdown(curve, check, bias_summary, table, elapsed_seconds, baseline_text)
    full_text = text + "\n" + forecast_text
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(full_text)
    print(full_text)
    print("Pre-registered forecast predictions (PREDICTIONS.md):")
    for verdict in forecast_summary["predictions"]:
        word = forecast_experiment.verdict_word(verdict["passed"])
        print(f"  {verdict['id']} {word}: {verdict['claim']} (observed {verdict['observed']})")
    print(
        f"figures: {hero_path}, {bias_path}; forecast files: {args.forecast_outdir}; "
        f"baseline files: {args.baseline_outdir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
