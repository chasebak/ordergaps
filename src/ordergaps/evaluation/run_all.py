"""Run the three evaluation experiments with frozen seeds and write the
figures and the results tables.

    python -m ordergaps.evaluation.run_all              # writes under out/
    python -m ordergaps.evaluation.run_all --outdir figures   # committed figures

Seeds, shares and the simulator regime below were frozen BEFORE the
committed results were generated; see README "Design decisions".
"""

import argparse
import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ordergaps.config import CAVEAT
from ordergaps.evaluation.experiments import (
    acceptance_check,
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


def results_markdown(curve, check, bias_summary, table, elapsed_seconds):
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
    lines.append(f"## 3. Counted vs allocated (vendor share {table['table_share']:.2f})")
    lines.append("")
    lines.append("| quantity | kind | error |")
    lines.append("|---|---|---|")
    lines.append(
        f"| window totals between seen orders | counted (ID arithmetic) | "
        f"exact: {table['n_windows_exact']}/{table['n_windows']} windows match ground truth |"
    )
    lines.append(
        f"| vendor share of orders, fully-pinned days | counted (sampling noise only) | "
        f"mean abs deviation from true share: {table['full_day_share_mean_abs_err']:.4f} |"
    )
    lines.append(
        f"| per-day total volume, full days | counted (boundary spill only) | "
        f"median rel err {table['full_day_err_median']:.2%} |"
    )
    lines.append(
        f"| weekly in-store volume (channel split) | allocated (model) | "
        f"median rel err {table['channel_err_median']:.2%} "
        f"[IQR {table['channel_err_p25']:.2%}, {table['channel_err_p75']:.2%}] |"
    )
    lines.append(
        f"| per-day total volume, partial/spread days | allocated (timing model) | "
        f"median rel err {table['nonfull_day_err_median']:.2%} "
        f"[IQR {table['nonfull_day_err_p25']:.2%}, {table['nonfull_day_err_p75']:.2%}] |"
    )
    lines.append("")
    lines.append(f"Generated in {elapsed_seconds:.1f} s by `python -m ordergaps.evaluation.run_all`")
    lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default=os.path.join("out", "figures"))
    parser.add_argument("--report", default=os.path.join("out", "results.md"))
    args = parser.parse_args(argv)

    start_time = time.monotonic()
    base = frozen_base_config()
    records = run_matrix(base, SHARES, FROZEN_SEEDS)
    curve = error_curve(records)
    check = acceptance_check(curve)
    bias_summary = length_bias_summary(records)
    table = counted_vs_allocated(records, TABLE_SHARE)
    elapsed_seconds = time.monotonic() - start_time

    os.makedirs(args.outdir, exist_ok=True)
    report_dir = os.path.dirname(args.report)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    hero_path = os.path.join(args.outdir, "hero_error_vs_fraction.png")
    bias_path = os.path.join(args.outdir, "length_bias.png")
    hero_figure(records, curve, check, hero_path)
    length_bias_figure(records, bias_summary, bias_path)

    text = results_markdown(curve, check, bias_summary, table, elapsed_seconds)
    with open(args.report, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(text)
    print(f"figures: {hero_path}, {bias_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
