"""The committed forecast outputs (results/) are what the tested code makes.

Chain checked here: code -> folds CSV (spot-recomputed for two (share, seed)
slices, a second or two) -> summary JSON and forecast.md (recomputed in full
from the committed CSVs) -> README (tests/test_readme_drift.py).
The full recompute (`python -m ordergaps.evaluation.forecast_experiment
--check`, about 25 s) is run locally before a commit, not in CI.

Also: a key allowlist for the JSON and a column allowlist for the CSVs (the
outputs carry scores and counts only: no order numbers, baskets or dollars),
and a size budget.
"""

import json
from pathlib import Path

import pytest

from ordergaps.evaluation import forecast_experiment as fx

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
SIZE_BUDGET_BYTES = 300_000
SUMMARY_KEYS = {
    "by_share",
    "diagnostics",
    "headline",
    "owner",
    "predictions",
    "readme_headline",
    "setup",
    "tables",
    "why_this_metric",
}
FOLD_COLUMNS = [
    "share", "seed", "origin", "series", "model", "n_days", "sum_abs_err", "sum_actual",
    "mae", "scale", "mase", "n_in80", "n_in95", "sum_width80", "n_zero_actual",
    "achievable80_sum", "achievable95_sum", "achievable80_var", "achievable95_var",
]
ORIGIN_COLUMNS = [
    "share", "seed", "origin", "n_train_days", "n_full_train_days", "n_reset_span_days",
    "dropped_day", "dropped_est", "dropped_true", "leak_days_compared", "leak_days_changed",
    "leak_rates_compared", "leak_rates_changed",
]


def read_committed(name):
    return (RESULTS_DIR / name).read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def committed():
    names = (fx.FOLDS_FILE, fx.ORIGINS_FILE, fx.SUMMARY_FILE, fx.MARKDOWN_FILE)
    texts = {}
    for name in names:
        texts[name] = read_committed(name)
    return texts


def test_summary_and_markdown_follow_from_the_committed_csvs(committed):
    rebuilt = fx.outputs_from_csv(committed[fx.FOLDS_FILE], committed[fx.ORIGINS_FILE])
    assert rebuilt[fx.MARKDOWN_FILE] == committed[fx.MARKDOWN_FILE]
    rebuilt_summary = json.loads(rebuilt[fx.SUMMARY_FILE])
    committed_summary = json.loads(committed[fx.SUMMARY_FILE])
    assert fx.structures_close(rebuilt_summary, committed_summary)


def slice_csv_text(csv_text, share, seed):
    frame = fx.csv_text_to_frame(csv_text)
    mask = (frame["share"] == share) & (frame["seed"] == seed)
    selected = frame[mask].reset_index(drop=True)
    return fx.frame_to_csv_text(selected)


@pytest.mark.parametrize("share, seed", [(0.10, 1101), (0.02, 1808)])
def test_committed_folds_match_a_fresh_run_of_one_slice(committed, share, seed):
    folds, origins = fx.run_forecast_experiment(shares=(share,), seeds=(seed,))
    fresh_folds = fx.frame_to_csv_text(folds)
    fresh_origins = fx.frame_to_csv_text(origins)
    committed_folds = slice_csv_text(committed[fx.FOLDS_FILE], share, seed)
    committed_origins = slice_csv_text(committed[fx.ORIGINS_FILE], share, seed)
    assert fx.frames_close(fresh_folds, committed_folds)
    assert fx.frames_close(fresh_origins, committed_origins)


def test_control_slice_check_catches_a_changed_number(committed):
    folds_text = committed[fx.FOLDS_FILE]
    frame = fx.csv_text_to_frame(folds_text)
    frame.loc[0, "sum_abs_err"] = frame.loc[0, "sum_abs_err"] + 0.5
    altered = fx.frame_to_csv_text(frame)
    assert not fx.frames_close(altered, folds_text)


def test_fold_count_is_the_pre_registered_one(committed):
    summary = json.loads(committed[fx.SUMMARY_FILE])
    setup = summary["setup"]
    # PREDICTIONS.md: 6 origins per seed, 48 folds, 336 test days per series.
    assert setup["n_origins_per_seed"] == 6
    assert setup["n_folds"] == 48
    assert setup["n_test_days"] == 336
    assert setup["seeds"] == [1101, 1202, 1303, 1404, 1505, 1606, 1707, 1808]


def test_every_prediction_prints_a_verdict(committed):
    summary = json.loads(committed[fx.SUMMARY_FILE])
    ids = []
    for verdict in summary["predictions"]:
        ids.append(verdict["id"])
    assert ids == ["P1", "P2", "P3", "P4", "P5", "B1"]
    lines = fx.verdict_lines(summary)
    for line in lines:
        assert line.endswith(("-> PASS", "-> FAIL"))
    for line in lines:
        assert line in committed[fx.MARKDOWN_FILE]


def test_outputs_carry_only_allowlisted_keys_and_columns(committed):
    summary = json.loads(committed[fx.SUMMARY_FILE])
    assert set(summary) == SUMMARY_KEYS
    folds = fx.csv_text_to_frame(committed[fx.FOLDS_FILE])
    origins = fx.csv_text_to_frame(committed[fx.ORIGINS_FILE])
    assert list(folds.columns) == FOLD_COLUMNS
    assert list(origins.columns) == ORIGIN_COLUMNS


def test_outputs_fit_the_size_budget():
    total = 0
    paths = sorted(RESULTS_DIR.iterdir())
    for path in paths:
        total += path.stat().st_size
    assert total <= SIZE_BUDGET_BYTES


def test_why_this_metric_is_one_shared_copy(committed):
    summary = json.loads(committed[fx.SUMMARY_FILE])
    assert summary["why_this_metric"] == fx.WHY_THIS_METRIC
    why_keys = sorted(fx.WHY_THIS_METRIC)
    for key in why_keys:
        box_lines = fx.why_lines(key)
        for line in box_lines:
            assert line in committed[fx.MARKDOWN_FILE]
