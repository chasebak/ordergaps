"""Every number in DECISIONS.md is copied, not typed.

Why: DECISIONS.md quotes measurements ("alternative Z lost by N"). Those
numbers must come from a committed generated file, so each number token in
it must appear in results/forecast.md (README section 4, written by tested
code), in results/baseline_yield.md (README sections 1.1 and 1.2, written by
run_all.baseline_outputs), in the Gap Explorer's summary and meta files
docs/data/summary.json and meta.json (written by ordergaps.viz, checked by
tests/test_viz.py), in the few named week.json keys DECISIONS cites
(WEEK_KEYS) or in PREDICTIONS.md (the pre-registered design constants:
seeds, windows, thresholds). Dated heading lines are skipped.

Why not all of week.json and share_grid.json: their per-day and per-gap
values are about a thousand numbers, so nearly any small integer or
two-decimal value would pass whatever its source.

Failure mode guarded: a number computed by hand in prose ("about four
times", a rounded ratio) or left stale after a rerun. A control plants a
typed number and requires the check to catch it.
"""

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?%?")
# week.json keys DECISIONS.md cites (the week shown, the gap counts).
WEEK_KEYS = ("iso_year", "iso_week", "seed")
WEEK_SHARE_KEYS = ("n_gaps_touching_week",)


def number_tokens(text):
    return NUMBER_PATTERN.findall(text)


def allowed_tokens():
    sources = (
        REPO_ROOT / "results" / "forecast.md",
        REPO_ROOT / "results" / "baseline_yield.md",
        REPO_ROOT / "docs" / "data" / "summary.json",
        REPO_ROOT / "docs" / "data" / "meta.json",
        REPO_ROOT / "PREDICTIONS.md",
    )
    tokens = set()
    for path in sources:
        text = path.read_text(encoding="utf-8")
        tokens.update(number_tokens(text))
    week_tokens = cited_week_tokens(REPO_ROOT / "docs" / "data" / "week.json")
    tokens.update(week_tokens)
    return tokens


def cited_week_tokens(path):
    """Number tokens of the WEEK_KEYS values and of each share's
    WEEK_SHARE_KEYS values in week.json, and nothing else from that file."""
    week = json.loads(path.read_text(encoding="utf-8"))
    tokens = set()
    for key in WEEK_KEYS:
        tokens.update(number_tokens(str(week[key])))
    share_entries = week["by_share"]
    for entry in share_entries:
        for key in WEEK_SHARE_KEYS:
            tokens.update(number_tokens(str(entry[key])))
    return tokens


def untraced_numbers(decisions_text, allowed):
    """'line N: token' for each number not found in the allowed sources."""
    problems = []
    lines = decisions_text.splitlines()
    numbered = enumerate(lines, start=1)
    for line_number, line in numbered:
        if line.startswith("#"):
            continue
        tokens = number_tokens(line)
        for token in tokens:
            if token not in allowed:
                problems.append(f"line {line_number}: {token}")
    return problems


def test_every_decisions_number_is_copied():
    text = (REPO_ROOT / "DECISIONS.md").read_text(encoding="utf-8")
    problems = untraced_numbers(text, allowed_tokens())
    assert problems == []


def test_control_a_per_day_page_value_is_no_longer_allowed():
    # Control: a value that only the per-day week data holds (a day's
    # estimated total) must not pass just because the page prints it.
    week = json.loads((REPO_ROOT / "docs" / "data" / "week.json").read_text(encoding="utf-8"))
    allowed = allowed_tokens()
    found_one = False
    share_entries = week["by_share"]
    for entry in share_entries:
        day_entries = entry["days"]
        for day in day_entries:
            token = str(day["est_total"])
            if token not in allowed:
                found_one = True
    assert found_one


def test_control_a_typed_number_is_caught():
    planted = "the weekday average wins by 3.7779 orders a day"
    problems = untraced_numbers(planted, allowed_tokens())
    assert problems == ["line 1: 3.7779"]
