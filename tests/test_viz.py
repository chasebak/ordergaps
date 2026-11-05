"""The Gap Explorer page data (docs/data/*.json) against the functions behind
the README.

Why: the page computes nothing, so every check on its numbers lives here.
- Freshness: the committed files equal a build from the frozen matrix (the
  session's frozen_records fixture, so CI does not pay for a second matrix;
  `python -m ordergaps.viz build --check` is the local full rebuild).
- Parity: summary.json's per-share rows equal error_curve /
  baseline_and_yield / length_bias_summary / counted_vs_allocated at the
  page's rounding; share_grid.json equals run_matrix()'s rows; each week
  entry's score equals its share_grid cell, so sim and result belong to the
  same run; the page's section-3 rows at vendor share 0.10 are the README's.
- Logic: every day conserves seen + allocated = estimated total; every
  listed gap has unseen = n2 - n1 - 1 = the simulator's count; the chosen
  week is interior (fully covered, no reset).
- Privacy: the build reads no file; the bundle carries no basket or money
  keys, no list as long as the data rows, and stays under 300 KB.
- Determinism: week.json is identical under TZ=UTC and another time zone.
Each logic and privacy check has a control that must fail.
"""

import builtins
import copy
import io
import json
import math
import os
import pathlib
import subprocess
import sys

import pytest

from ordergaps import viz
from ordergaps.evaluation.experiments import (
    baseline_and_yield,
    counted_vs_allocated,
    error_curve,
    length_bias_summary,
)
from ordergaps.evaluation.run_all import FROZEN_SEEDS, SHARES, TABLE_SHARE, counted_rows

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
# Exactly the tag in the portfolio page kit's CONVENTIONS.md (typed here, not
# imported from viz, so a typo in viz.CHART_JS_TAG cannot pass itself).
CONVENTIONS_TAG = (
    '<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js" '
    'integrity="sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6djDQZjh3Y8SZ1d+6I+1zze6Z7k'
    'HXO7q3UyZAWw==" crossorigin="anonymous" referrerpolicy="no-referrer"></script>'
)
CONVENTIONS_FOOTER = (
    "Every number was computed by this repo's tested Python and copied; the page computes nothing."
)
OTHER_TIME_ZONE = "Asia/Tokyo"


@pytest.fixture(scope="module")
def page_outputs(frozen_records):
    return viz.build_outputs(records=frozen_records)


@pytest.fixture(scope="module")
def bundle(page_outputs):
    parsed = {}
    paths = sorted(page_outputs)
    for path in paths:
        parsed[path] = json.loads(page_outputs[path])
    return parsed


def rounded(value):
    """Function output rounded and serialized exactly like the page JSON."""
    return json.loads(json.dumps(viz.page_value(value), sort_keys=True))


# ---------- freshness and parity ----------


def test_committed_page_data_are_fresh(page_outputs):
    problems = viz.check(outputs=page_outputs)
    assert problems == []


def test_summary_rows_equal_the_readme_functions(bundle, frozen_records):
    summary = bundle[viz.SUMMARY_PATH]
    curve = error_curve(frozen_records)
    baseline = baseline_and_yield(frozen_records)
    bias = length_bias_summary(frozen_records)
    assert len(summary["by_share"]) == len(SHARES)
    curve_rows = curve.to_dict("records")
    baseline_rows = baseline.to_dict("records")
    bias_rows = bias.to_dict("records")
    share_indexes = range(len(SHARES))
    for index in share_indexes:
        row = summary["by_share"][index]
        share = SHARES[index]
        assert row["share"] == share
        assert row["median_weekly_err"] == rounded(curve_rows[index]["median_weekly_err"])
        assert row["mean_fraction_observed"] == rounded(curve_rows[index]["mean_fraction_observed"])
        assert row["uniform_weekly_err"] == rounded(baseline_rows[index]["uniform_weekly_err"])
        assert row["verdict"] == baseline_rows[index]["verdict"]
        assert row["n_full_days"] == baseline_rows[index]["n_full_days"]
        assert row["full_day_yield"] == rounded(baseline_rows[index]["full_day_yield"])
        assert row["mean_bias"] == rounded(bias_rows[index]["mean_bias"])
        assert row["counted"] == rounded(counted_vs_allocated(frozen_records, share))


def test_summary_baseline_rows_equal_the_committed_baseline_json(bundle):
    summary = bundle[viz.SUMMARY_PATH]
    committed_path = REPO_ROOT / "results" / "baseline_yield.json"
    committed = json.loads(committed_path.read_text(encoding="utf-8"))
    pairs = zip(summary["by_share"], committed["rows"])
    for page_row, committed_row in pairs:
        assert page_row["uniform_weekly_err"] == committed_row["uniform_weekly_err"]
        assert page_row["median_weekly_err"] == committed_row["model_weekly_err"]
        assert page_row["n_open_days"] == committed_row["n_open_days"]


def test_page_section_three_rows_are_the_readme_rows(bundle, frozen_records):
    """At vendor share 0.10 the page's table prints the README's lines."""
    summary = bundle[viz.SUMMARY_PATH]
    readme_text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    table_index = SHARES.index(TABLE_SHARE)
    page_rows = summary["by_share"][table_index]["counted_rows"]
    expected = counted_rows(counted_vs_allocated(frozen_records, TABLE_SHARE))
    assert len(page_rows) == len(expected)
    pairs = zip(page_rows, expected)
    for page_row, (quantity, kind, error_text) in pairs:
        assert (page_row["quantity"], page_row["kind"], page_row["error"]) == (
            quantity,
            kind,
            error_text,
        )
        readme_line = f"| {quantity} | {kind} | {error_text} |"
        assert readme_line in readme_text
    windows_text = summary["by_share"][table_index]["text"]["windows"]
    assert windows_text == "13,713 / 13,713"


def test_share_grid_equals_run_matrix(bundle, frozen_records):
    grid = bundle[viz.GRID_PATH]
    records = frozen_records.to_dict("records")
    assert len(grid["cells"]) == len(SHARES) * len(FROZEN_SEEDS)
    pairs = zip(grid["cells"], records)
    for cell, record in pairs:
        expected = {}
        for field in viz.GRID_FIELDS:
            expected[field] = record[field]
        assert cell == rounded(expected)


def test_week_scores_equal_their_grid_cells(bundle):
    """The week view's sim and result are the same run as share_grid's seed-101 cell."""
    grid = bundle[viz.GRID_PATH]
    week = bundle[viz.WEEK_PATH]
    cells_by_key = {}
    for cell in grid["cells"]:
        cells_by_key[(cell["share"], cell["seed"])] = cell
    for entry in week["by_share"]:
        cell = cells_by_key[(entry["share"], week["seed"])]
        score = entry["score"]
        score_keys = sorted(score)
        for key in score_keys:
            assert score[key] == cell[key], key
        assert entry["fraction_observed"] == cell["fraction_observed"]


# ---------- logic checks ----------


def test_week_is_interior_and_before_the_reset(bundle):
    week = bundle[viz.WEEK_PATH]
    assert (week["iso_year"], week["iso_week"]) == (2025, 12)
    for entry in week["by_share"]:
        assert entry["fully_covered"] is True
        assert entry["has_anomaly"] is False
        assert len(entry["days"]) == 7
        for day in entry["days"]:
            assert day["covered"] is True


def conservation_problems(days):
    """'day: rule' for each day whose parts do not add up (relative 1e-9).

    Rules: seen = seen_register + seen_online; allocated = unseen_in_store +
    unseen_online; seen + allocated = est_total; est_in_store =
    seen_register + unseen_in_store.
    """
    problems = []
    for day in days:
        checks = (
            ("seen", day["seen_register"] + day["seen_online"], day["seen"]),
            ("allocated", day["unseen_in_store"] + day["unseen_online"], day["allocated"]),
            ("total", day["seen"] + day["allocated"], day["est_total"]),
            ("in_store", day["seen_register"] + day["unseen_in_store"], day["est_in_store"]),
        )
        for name, parts, whole in checks:
            if not math.isclose(parts, whole, rel_tol=1e-9, abs_tol=1e-9):
                problems.append(f"{day['day']}: {name}")
    return problems


def test_every_day_conserves_seen_plus_allocated(bundle):
    week = bundle[viz.WEEK_PATH]
    for entry in week["by_share"]:
        assert conservation_problems(entry["days"]) == []


def test_control_conservation_catches_a_lost_order(bundle):
    week = bundle[viz.WEEK_PATH]
    days = copy.deepcopy(week["by_share"][0]["days"])
    days[3]["allocated"] = days[3]["allocated"] - 1.0
    assert conservation_problems(days) == [
        f"{days[3]['day']}: allocated",
        f"{days[3]['day']}: total",
    ]


def gap_problems(gaps):
    """'n1-n2: rule' for each listed gap breaking the counted identity.

    Rules: unseen = n2 - n1 - 1; the simulator created exactly that many
    orders strictly between t1 and t2; the modelled split hands out exactly
    unseen orders.
    """
    problems = []
    for gap in gaps:
        key = f"{gap['n1']}-{gap['n2']}"
        if gap["unseen"] != gap["n2"] - gap["n1"] - 1:
            problems.append(f"{key}: arithmetic")
        if gap["truth_between"] != gap["unseen"]:
            problems.append(f"{key}: truth")
        split_total = 0.0
        for part in gap["split"]:
            split_total += part["in_store"] + part["online"]
        if not math.isclose(split_total, gap["unseen"], rel_tol=1e-9, abs_tol=1e-9):
            problems.append(f"{key}: split")
    return problems


def test_every_listed_gap_is_exact_against_the_simulator(bundle):
    week = bundle[viz.WEEK_PATH]
    for entry in week["by_share"]:
        assert len(entry["gaps"]) >= 6
        assert gap_problems(entry["gaps"]) == []


def test_control_gap_check_catches_an_off_by_one(bundle):
    week = bundle[viz.WEEK_PATH]
    gaps = copy.deepcopy(week["by_share"][0]["gaps"][:1])
    gaps[0]["truth_between"] = gaps[0]["truth_between"] + 1
    key = f"{gaps[0]['n1']}-{gaps[0]['n2']}"
    assert gap_problems(gaps) == [f"{key}: truth"]


def test_more_share_means_more_full_days_and_nested_orders(bundle):
    """The slider's story: the same week becomes countable as the share grows."""
    week = bundle[viz.WEEK_PATH]
    entries = week["by_share"]
    previous_observed = 0
    for entry in entries:
        assert entry["n_observed"] > previous_observed
        previous_observed = entry["n_observed"]
    first_full = 0
    for day in entries[0]["days"]:
        if day["label"] == "full":
            first_full += 1
    last_full = 0
    for day in entries[-1]["days"]:
        if day["label"] == "full":
            last_full += 1
    assert last_full > first_full


# ---------- privacy, size, static files ----------


def test_bundle_passes_allowlist_banned_keys_list_length_and_size(page_outputs):
    assert viz.bundle_problems(page_outputs) == []
    total = 0
    texts = list(page_outputs.values())
    for text in texts:
        total += len(text.encode("utf-8"))
    assert total <= 300_000


def test_control_bundle_check_catches_basket_keys_long_lists_and_size(page_outputs):
    week = json.loads(page_outputs[viz.WEEK_PATH])
    week["by_share"][0]["days"][0]["total"] = 12.5
    week["by_share"][0]["days"][0]["vendor_revenue"] = 99.0
    week["by_share"][0]["gaps"] = list(range(viz.MAX_LIST_LENGTH + 1))
    planted = dict(page_outputs)
    planted[viz.WEEK_PATH] = json.dumps(week)
    planted[viz.SUMMARY_PATH] = page_outputs[viz.SUMMARY_PATH] + " " * viz.DATA_BUDGET_BYTES
    problems = viz.bundle_problems(planted)
    joined = "\n".join(problems)
    assert "keys outside the allowlist" in joined
    assert "basket or money keys: ['total', 'vendor_revenue']" in joined
    assert f"a list of {viz.MAX_LIST_LENGTH + 1} entries" in joined
    assert "budget" in joined


def test_the_build_reads_no_file(monkeypatch, frozen_records):
    """The build can only call the simulator: any file open or read fails it."""

    def refuse(*args, **kwargs):
        raise AssertionError("the page build tried to read a file")

    monkeypatch.setattr(builtins, "open", refuse)
    monkeypatch.setattr(io, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "open", refuse)
    monkeypatch.setattr(pathlib.Path, "read_text", refuse)
    monkeypatch.setattr(pathlib.Path, "read_bytes", refuse)
    outputs = viz.build_outputs(records=frozen_records)
    assert sorted(outputs) == sorted(viz.DATA_PATHS)


def test_control_the_no_read_guard_catches_a_read(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("read")

    monkeypatch.setattr(pathlib.Path, "read_text", refuse)
    with pytest.raises(AssertionError):
        (REPO_ROOT / "README.md").read_text(encoding="utf-8")


def test_the_cli_takes_no_input_file():
    with pytest.raises(SystemExit):
        viz.main(["build", "--input", "sales.csv"])


def test_static_files_tokens_nojekyll_and_chart_tag():
    assert viz.static_problems(REPO_ROOT) == []
    assert viz.CHART_JS_TAG == CONVENTIONS_TAG
    assert viz.FOOTER == CONVENTIONS_FOOTER
    index_text = (REPO_ROOT / viz.INDEX_PATH).read_text(encoding="utf-8")
    assert index_text.count(CONVENTIONS_TAG) == 1
    tokens = (REPO_ROOT / viz.TOKENS_PATH).read_bytes()
    assert viz.sha256_base64(tokens) == viz.TOKENS_SHA256_B64
    assert (REPO_ROOT / viz.NOJEKYLL_PATH).is_file()


def test_control_static_check_catches_an_edited_tokens_copy(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    tokens = (REPO_ROOT / viz.TOKENS_PATH).read_bytes()
    (docs / "tokens.css").write_bytes(tokens + b"\n")
    (docs / "index.html").write_text("<html></html>", encoding="utf-8")
    problems = viz.static_problems(tmp_path)
    assert f"{viz.TOKENS_PATH} is not the page kit's tokens.css" in problems
    assert f"{viz.NOJEKYLL_PATH} is missing" in problems
    assert f"{viz.INDEX_PATH} lacks the pinned Chart.js tag" in problems


def test_control_check_catches_a_stale_file(page_outputs):
    stale = dict(page_outputs)
    stale[viz.WEEK_PATH] = page_outputs[viz.WEEK_PATH].replace('"seed": 101', '"seed": 102', 1)
    assert stale[viz.WEEK_PATH] != page_outputs[viz.WEEK_PATH]
    problems = viz.check(outputs=stale)
    assert f"{viz.WEEK_PATH} is stale (run {viz.GENERATED_BY})" in problems


def test_meta_records_the_measured_sizes(bundle, page_outputs):
    meta = bundle[viz.META_PATH]
    data_bytes = meta["data_bytes"]
    data_names = (viz.SUMMARY_PATH, viz.GRID_PATH, viz.WEEK_PATH)
    for path in data_names:
        name = path.rsplit("/", 1)[1]
        assert data_bytes[name] == len(page_outputs[path].encode("utf-8"))
    assert meta["data_budget_bytes"] == 300_000


# ---------- determinism ----------


def week_text_under(time_zone):
    environment = dict(os.environ)
    environment["TZ"] = time_zone
    code = "import sys; from ordergaps import viz; sys.stdout.write(viz.week_json_text())"
    return subprocess.Popen(
        [sys.executable, "-c", code],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_week_json_is_deterministic_under_tz_utc_and_another_zone():
    committed = (REPO_ROOT / viz.WEEK_PATH).read_text(encoding="utf-8")
    utc_process = week_text_under("UTC")
    other_process = week_text_under(OTHER_TIME_ZONE)
    utc_text, utc_errors = utc_process.communicate(timeout=120)
    other_text, other_errors = other_process.communicate(timeout=120)
    assert utc_process.returncode == 0, utc_errors
    assert other_process.returncode == 0, other_errors
    assert utc_text == committed
    assert other_text == committed
