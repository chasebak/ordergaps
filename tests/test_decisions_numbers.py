"""Every number in DECISIONS.md is copied, not typed.

Why: DECISIONS.md quotes measurements ("alternative Z lost by N"). Those
numbers must come from a committed generated file, so each number token in
it must appear in results/forecast.md (README section 4, written by tested
code), in results/baseline_yield.md (README sections 1.1 and 1.2, written by
run_all.baseline_outputs) or in PREDICTIONS.md (the pre-registered design
constants: seeds, windows, thresholds). Dated heading lines are skipped.

Failure mode guarded: a number computed by hand in prose ("about four
times", a rounded ratio) or left stale after a rerun. A control plants a
typed number and requires the check to catch it.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?%?")


def number_tokens(text):
    return NUMBER_PATTERN.findall(text)


def allowed_tokens():
    sources = (
        REPO_ROOT / "results" / "forecast.md",
        REPO_ROOT / "results" / "baseline_yield.md",
        REPO_ROOT / "PREDICTIONS.md",
    )
    tokens = set()
    for path in sources:
        text = path.read_text(encoding="utf-8")
        tokens.update(number_tokens(text))
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


def test_control_a_typed_number_is_caught():
    planted = "the weekday average wins by 3.7779 orders a day"
    problems = untraced_numbers(planted, allowed_tokens())
    assert problems == ["line 1: 3.7779"]
