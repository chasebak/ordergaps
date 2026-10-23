"""Demo: simulate a store, estimate its volume from one vendor's sales,
and write out/report.md plus figures — all under untracked out/.

    python -m ordergaps.demo
"""

import os
import time

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ordergaps.config import CAVEAT, WEEKDAY_NAMES
from ordergaps.estimator import estimate
from ordergaps.simulator import SimConfig, simulate

DEMO_SEED = 7


def demo_config():
    return SimConfig(
        n_weeks=8,
        vendor_share=0.15,
        void_rate=0.02,
        reset_week=4,
        drift_per_week=0.04,
    )


def daily_figure(result, sim, path):
    merged = result.daily.merge(sim.truth_daily, on="day", suffixes=("", "_true"))
    figure, axes = plt.subplots(figsize=(9.0, 5.0), dpi=100)
    axes.plot(merged["day"], merged["total"], color="#444444", linewidth=1.2, label="truth")
    axes.plot(
        merged["day"],
        merged["est_total"],
        color="#1f4e79",
        linewidth=1.6,
        linestyle="--",
        label="estimate",
    )
    axes.set_xlabel("day")
    axes.set_ylabel("orders created per day")
    axes.set_title("Daily store volume: estimate vs simulator ground truth")
    axes.grid(True, alpha=0.25)
    axes.legend(loc="upper left")
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def weekly_figure(result, sim, path):
    merged = result.weekly.merge(
        sim.truth_weekly, on=["iso_year", "iso_week"], suffixes=("", "_true")
    )
    labels = []
    truth_positions = []
    estimate_positions = []
    positions = []
    numbered_rows = list(enumerate(merged.itertuples(index=False)))
    for position, row in numbered_rows:
        labels.append(f"{row.iso_year}-W{row.iso_week:02d}")
        positions.append(position)
        truth_positions.append(position - 0.2)
        estimate_positions.append(position + 0.2)
    figure, axes = plt.subplots(figsize=(9.0, 5.0), dpi=100)
    axes.bar(truth_positions, merged["total"], width=0.4, color="#444444", label="truth")
    axes.bar(
        estimate_positions,
        merged["est_total"],
        width=0.4,
        color="#1f4e79",
        label="estimate",
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(labels, rotation=30)
    axes.set_ylabel("orders created per week")
    axes.set_title("Weekly store volume: estimate vs simulator ground truth")
    axes.grid(True, axis="y", alpha=0.25)
    axes.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def weekly_markdown(result, sim):
    merged = result.weekly.merge(
        sim.truth_weekly, on=["iso_year", "iso_week"], suffixes=("", "_true")
    )
    lines = []
    lines.append(
        "| week | est total | true total | rel err | est in-store | true in-store "
        "| vendor share of in-store | flags |"
    )
    lines.append("|---|---|---|---|---|---|---|---|")
    merged_tuples = list(merged.itertuples(index=False))
    for row in merged_tuples:
        if row.total > 0:
            rel_err = f"{abs(row.est_total - row.total) / row.total:.2%}"
        else:
            rel_err = "-"
        flags = []
        if not row.fully_covered:
            flags.append("edge week")
        if row.has_anomaly:
            flags.append("anomaly")
        flag_text = ", ".join(flags) if flags else ""
        lines.append(
            f"| {row.iso_year}-W{row.iso_week:02d} | {row.est_total:.0f} | {row.total} "
            f"| {rel_err} | {row.est_in_store:.0f} | {row.in_store} "
            f"| {row.vendor_share_in_store:.1%} | {flag_text} |"
        )
    return "\n".join(lines)


def rates_markdown(result, sim):
    lines = []
    lines.append("| weekday | learned rate (orders/open minute) | true rate | source |")
    lines.append("|---|---|---|---|")
    weekday_values = range(7)
    for weekday in weekday_values:
        true_rate = sim.true_weekday_rates.get(weekday)
        if true_rate is None:
            continue
        learned_rate = result.rates.rate_for(weekday)
        source = result.rates.source.get(weekday, "pooled")
        lines.append(
            f"| {WEEKDAY_NAMES[weekday]} | {learned_rate:.3f} | {true_rate:.3f} | {source} |"
        )
    return "\n".join(lines)


def main():
    start_time = time.monotonic()
    os.makedirs("out", exist_ok=True)

    config = demo_config()
    sim = simulate(config, DEMO_SEED)
    result = estimate(sim.observed, config.store_hours)

    daily_path = os.path.join("out", "daily_volume.png")
    daily_figure(result, sim, daily_path)
    weekly_path = os.path.join("out", "weekly_volume.png")
    weekly_figure(result, sim, weekly_path)

    label_counts = {}
    label_values = result.day_labels.values()
    for label in label_values:
        label_counts[label] = label_counts.get(label, 0) + 1

    lines = []
    lines.append("# ordergaps demo report")
    lines.append("")
    lines.append(
        f"Simulated {config.n_weeks} weeks of a store; the vendor saw "
        f"{len(sim.observed)} of {len(sim.orders)} orders "
        f"({sim.fraction_observed:.1%}). Planted pathologies: void orders "
        f"(rate {config.void_rate}), a numbering reset in week "
        f"{config.reset_week}, rate drift {config.drift_per_week:+.0%}/week."
    )
    lines.append("")
    lines.append(f"Caveat: {CAVEAT}")
    lines.append("")
    lines.append("## Weekly store volume, estimate vs ground truth")
    lines.append("")
    lines.append(weekly_markdown(result, sim))
    lines.append("")
    lines.append("## Learned weekday pace vs truth")
    lines.append("")
    lines.append(rates_markdown(result, sim))
    lines.append("")
    lines.append("## Day classification")
    lines.append("")
    label_items = sorted(label_counts.items())
    for label, count in label_items:
        lines.append(f"- {label}: {count} days")
    lines.append("")
    lines.append("## Dispersion check")
    lines.append("")
    lines.append(result.dispersion.describe())
    lines.append("")
    lines.append("## Anomalies detected")
    lines.append("")
    if result.anomalies:
        for anomaly in result.anomalies:
            lines.append(f"- {anomaly['kind']} at {anomaly['at']}: {anomaly['detail']}")
    else:
        lines.append("- none")
    lines.append("")
    lines.append(f"![daily volume]({os.path.basename(daily_path)})")
    lines.append("")
    lines.append(f"![weekly volume]({os.path.basename(weekly_path)})")
    lines.append("")
    elapsed_seconds = time.monotonic() - start_time
    lines.append(f"Demo finished in {elapsed_seconds:.1f} s.")
    lines.append("")

    report_path = os.path.join("out", "report.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))

    print(f"observed {len(sim.observed)} of {len(sim.orders)} orders "
          f"({sim.fraction_observed:.1%})")
    print(result.dispersion.describe())
    print(f"anomalies detected: {len(result.anomalies)}")
    print(f"wrote {report_path}, {daily_path} and {weekly_path}")
    print(f"demo finished in {elapsed_seconds:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
