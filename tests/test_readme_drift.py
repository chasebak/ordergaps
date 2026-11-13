"""README-number drift test: every result number the README quotes must be
exactly what the generator prints.

Why: the README's "Results" tables are copied by hand from the output of
`python -m ordergaps.evaluation.run_all` (out/results.md). out/ is not
committed, so the source of truth in CI is the generator itself:
run_all.results_markdown() fed by the frozen matrix for sections 1-3, and
the committed results/forecast.md for section 4 (itself checked against
the code by tests/test_forecast_outputs.py). If the code, the seeds or the
README change without the other, this test fails.

Sections 1.1, 1.2 and 1.3 (uniform baseline, full-day yield, estimation
KPI table with the window-exactness guarantee) are generated from the same
frozen matrix by run_all.baseline_outputs(); their README block
must equal the committed results/baseline_yield.md byte for byte, and that
file must equal a fresh generation (it holds the shared "why this metric"
text and the verdict sentence, which are not number-token checked).

Section 4 and the README's top headline are also checked as exact text:
the section-4 block must equal results/forecast.md byte for byte (it holds
the shared "why this metric" text, which must not be reworded on one side
only), and the two headline lines under the title must equal
forecast_summary.json "readme_headline".

The why boxes under tables 1, 2 and 3 must equal the shared
run_all.WHY_THIS_METRIC text (run_all.section_why_block), one box per
section.

Rule (what is checked):
- Only the README section "## Results ..." up to the next "## " heading.
- Inside it, only the lines the generator emits (sections are numbered
  "1."..."4.", "1.1"..."1.3" and "4.0"..."4.6"):
  1. every table row, keyed by (section number, first cell); the row's list
     of number tokens must equal the generated row's list, and every
     generated row must appear in the README (a dropped row is drift too);
  2. the "Acceptance band:" sentence (wrapped over lines in the README), up
     to its PASS/FAIL verdict: same number tokens, same verdict;
  3. the section headings' number tokens (e.g. "vendor share 0.10"), both
     ways: a README heading must match the generated one, and every
     generated section heading must appear in the README (a dropped
     section is drift too, same as a dropped row).
- Prose around the tables is NOT checked: it is hand-written, not generated
  (e.g. "at a 2% vendor share" names a config value, not a result).
Comparing number tokens, not whole lines, lets wording and column text be
edited freely while any changed digit, sign or % fails.

Failure modes guarded: a README number typed or rounded by hand, a stale
README after a generator/simulator change, and a vacuous pass (the test
also requires a minimum number of checked rows, and a control mutates one
README number and requires the check to catch it).
"""

import json
import re
from pathlib import Path

from ordergaps.evaluation.experiments import (
    acceptance_check,
    counted_vs_allocated,
    error_curve,
    length_bias_summary,
)
from ordergaps.evaluation.forecast_experiment import structures_close
from ordergaps.evaluation.run_all import (
    BASELINE_MARKDOWN_FILE,
    BASELINE_SUMMARY_FILE,
    SECTION_WHY_KEYS,
    TABLE_SHARE,
    baseline_outputs,
    results_markdown,
    section_why_block,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
README_PATH = REPO_ROOT / "README.md"
FORECAST_MARKDOWN_PATH = REPO_ROOT / "results" / "forecast.md"
FORECAST_SUMMARY_PATH = REPO_ROOT / "results" / "forecast_summary.json"
BASELINE_MARKDOWN_PATH = REPO_ROOT / "results" / BASELINE_MARKDOWN_FILE
BASELINE_SUMMARY_PATH = REPO_ROOT / "results" / BASELINE_SUMMARY_FILE
BASELINE_HEADING = "#### 1.1 "
SECTION_TWO_HEADING = "### 2. "
NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?%?")
# "### 3. Title" -> "3"; "#### 4.1 Title" -> "4.1".
SECTION_PATTERN = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)?)[.\s]")
VERDICT_PATTERN = re.compile(r"Acceptance band:.*?(PASS|FAIL)")
# Sections 1-3: 6 + 6 + 5 data rows plus 3 header rows; sections 1.1-1.3:
# 6 + 6 + 3 data rows plus 3 header rows; section 4: 4 KPI rows, 3 x 4 model
# rows, 6 share rows, 5 check rows plus 6 header rows. Fewer means the parser
# stopped seeing the tables and the test would pass vacuously.
MIN_CHECKED_ROWS = 71
SECTION_FOUR_HEADING = "### 4."


def committed_forecast_markdown():
    return FORECAST_MARKDOWN_PATH.read_text(encoding="utf-8")


def generated_results_text(records):
    """The exact text run_all writes to out/results.md, minus timing."""
    curve = error_curve(records)
    check = acceptance_check(curve)
    bias_summary = length_bias_summary(records)
    table = counted_vs_allocated(records, TABLE_SHARE)
    baseline_text = baseline_outputs(records)[BASELINE_MARKDOWN_FILE]
    text = results_markdown(
        curve, check, bias_summary, table, elapsed_seconds=0.0, baseline_text=baseline_text
    )
    return text + "\n" + committed_forecast_markdown()


def readme_results_section(readme_text):
    """Lines from the '## Results' heading up to the next level-2 heading."""
    lines = readme_text.splitlines()
    section = []
    inside = False
    for line in lines:
        if line.startswith("## "):
            if inside:
                break
            inside = line.startswith("## Results")
            continue
        if inside:
            section.append(line)
    return section


def number_tokens(text):
    tokens = NUMBER_PATTERN.findall(text)
    return tokens


def is_separator_row(line):
    stripped = line.replace("|", "").replace("-", "").strip()
    return stripped == ""


def parse_result_lines(lines):
    """(rows, headings, acceptance) from a results text.

    rows: {(section, first cell): number tokens of the row}
    headings: {section: number tokens of the heading}
    acceptance: (number tokens, verdict) of the acceptance sentence, or None.
    """
    rows = {}
    headings = {}
    acceptance = None
    section = None
    index = 0
    n_lines = len(lines)
    while index < n_lines:
        line = lines[index]
        heading_match = SECTION_PATTERN.match(line)
        if heading_match:
            section = heading_match.group(1)
            headings[section] = number_tokens(line)
        elif line.startswith("|") and not is_separator_row(line):
            cells = line.strip().strip("|").split("|")
            key = (section, cells[0].strip())
            rows[key] = number_tokens(line)
        elif line.startswith("Acceptance band:"):
            # The README wraps the sentence; join until the blank line.
            paragraph = [line]
            next_index = index + 1
            while next_index < n_lines and lines[next_index].strip() != "":
                paragraph.append(lines[next_index].strip())
                next_index += 1
            joined = " ".join(paragraph)
            verdict_match = VERDICT_PATTERN.search(joined)
            if verdict_match:
                sentence = verdict_match.group(0)
                acceptance = (number_tokens(sentence), verdict_match.group(1))
        index += 1
    return rows, headings, acceptance


def readme_drift(readme_text, generated_text):
    """List of human-readable mismatches; empty means no drift."""
    readme_lines = readme_results_section(readme_text)
    generated_lines = generated_text.splitlines()
    readme_rows, readme_headings, readme_acceptance = parse_result_lines(readme_lines)
    gen_rows, gen_headings, gen_acceptance = parse_result_lines(generated_lines)

    problems = []
    readme_row_items = readme_rows.items()
    for key, tokens in readme_row_items:
        if key not in gen_rows:
            problems.append(f"README row {key} is not in the generated results")
        elif tokens != gen_rows[key]:
            problems.append(f"row {key}: README {tokens} != generated {gen_rows[key]}")
    gen_row_keys = gen_rows.keys()
    for key in gen_row_keys:
        if key not in readme_rows:
            problems.append(f"generated row {key} is missing from the README")
    readme_heading_items = readme_headings.items()
    for section, tokens in readme_heading_items:
        generated_tokens = gen_headings.get(section)
        if tokens != generated_tokens:
            problems.append(f"heading {section}: README {tokens} != generated {generated_tokens}")
    gen_heading_sections = gen_headings.keys()
    for section in gen_heading_sections:
        if section not in readme_headings:
            problems.append(f"generated heading {section} is missing from the README")
    if readme_acceptance != gen_acceptance:
        problems.append(
            f"acceptance: README {readme_acceptance} != generated {gen_acceptance}"
        )
    if len(readme_rows) < MIN_CHECKED_ROWS:
        problems.append(f"only {len(readme_rows)} README result rows parsed")
    return problems


def test_readme_results_match_generator(frozen_records):
    readme_text = README_PATH.read_text(encoding="utf-8")
    generated_text = generated_results_text(frozen_records)
    problems = readme_drift(readme_text, generated_text)
    assert problems == []


def test_drift_check_catches_an_altered_number(frozen_records):
    # Control that must fail: change one quoted result in memory.
    readme_text = README_PATH.read_text(encoding="utf-8")
    generated_text = generated_results_text(frozen_records)
    altered_text = readme_text.replace("| 0.20 | 0.195 | 0.0676% |", "| 0.20 | 0.195 | 0.0677% |")
    assert altered_text != readme_text
    problems = readme_drift(altered_text, generated_text)
    assert len(problems) == 1
    assert "0.0677%" in problems[0]


def test_drift_check_catches_a_dropped_section_heading(frozen_records):
    # Control that must fail: the README loses section 3's heading line, so
    # its rows fold into section 2 and the generated heading 3 goes missing.
    readme_text = README_PATH.read_text(encoding="utf-8")
    generated_text = generated_results_text(frozen_records)
    heading_line = "### 3. Counted vs allocated (vendor share 0.10)\n"
    altered_text = readme_text.replace(heading_line, "")
    assert altered_text != readme_text
    problems = readme_drift(altered_text, generated_text)
    assert "generated heading 3 is missing from the README" in problems


def readme_section_four(readme_text):
    """README text from '### 4.' up to the blank line before the next
    level-2 heading (forecast.md ends in one newline; the README adds the
    blank separator line)."""
    start = readme_text.index(SECTION_FOUR_HEADING)
    rest = readme_text[start:]
    end = rest.find("\n## ")
    if end == -1:
        return rest
    return rest[:end]


def readme_headline_block(readme_text):
    """The bold headline paragraph and the owner's line right under it.

    They are two one-line paragraphs: the headline, a blank line, the owner's
    line. Returns [] when the headline line is missing.
    """
    lines = readme_text.splitlines()
    n_lines = len(lines)
    line_indexes = range(n_lines)
    for index in line_indexes:
        if lines[index].startswith("**Forecast error added by the partial view"):
            if index + 2 < n_lines and lines[index + 1] == "":
                return [lines[index], lines[index + 2]]
            return [lines[index]]
    return []


def test_readme_section_four_is_the_committed_forecast_markdown():
    readme_text = README_PATH.read_text(encoding="utf-8")
    section = readme_section_four(readme_text)
    assert section == committed_forecast_markdown()


def test_readme_headline_lines_match_the_summary():
    readme_text = README_PATH.read_text(encoding="utf-8")
    summary = json.loads(FORECAST_SUMMARY_PATH.read_text(encoding="utf-8"))
    assert readme_headline_block(readme_text) == summary["readme_headline"]
    # The headline comes first, right under the title line.
    lines = readme_text.splitlines()
    assert lines[2].startswith("**Forecast error added by the partial view")


def test_drift_check_catches_an_altered_forecast_number(frozen_records):
    # Control that must fail: change one section-4 number in memory.
    readme_text = README_PATH.read_text(encoding="utf-8")
    generated_text = generated_results_text(frozen_records)
    summary = json.loads(FORECAST_SUMMARY_PATH.read_text(encoding="utf-8"))
    added = summary["by_share"][0]["added"]
    quoted = f"| {added:+.2%} |"
    assert quoted in readme_text
    altered_text = readme_text.replace(quoted, "| +9.99% |", 1)
    problems = readme_drift(altered_text, generated_text)
    assert len(problems) == 1
    assert "+9.99%" in problems[0]
    assert readme_section_four(altered_text) != committed_forecast_markdown()


def readme_baseline_block(readme_text):
    """README text from the section-1.1 heading up to the section-2 heading
    (baseline_yield.md plus the blank separator line)."""
    start = readme_text.index(BASELINE_HEADING)
    end = readme_text.index(SECTION_TWO_HEADING, start)
    return readme_text[start:end]


def test_readme_baseline_block_is_the_committed_markdown():
    readme_text = README_PATH.read_text(encoding="utf-8")
    committed = BASELINE_MARKDOWN_PATH.read_text(encoding="utf-8")
    assert readme_baseline_block(readme_text) == committed + "\n"


def test_committed_baseline_outputs_are_fresh(frozen_records):
    fresh = baseline_outputs(frozen_records)
    committed_markdown = BASELINE_MARKDOWN_PATH.read_text(encoding="utf-8")
    assert committed_markdown == fresh[BASELINE_MARKDOWN_FILE]
    committed_summary = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    fresh_summary = json.loads(fresh[BASELINE_SUMMARY_FILE])
    assert structures_close(committed_summary, fresh_summary)


def test_drift_check_catches_an_altered_baseline_number(frozen_records):
    # Control that must fail: change one section-1.1 number in memory. The
    # uniform column is the baseline's own number, so it is the one picked.
    readme_text = README_PATH.read_text(encoding="utf-8")
    generated_text = generated_results_text(frozen_records)
    summary = json.loads(BASELINE_SUMMARY_PATH.read_text(encoding="utf-8"))
    first_row = summary["rows"][0]
    quoted = f"| {first_row['uniform_weekly_err']:.4%} |"
    assert readme_text.count(quoted) == 1
    altered_text = readme_text.replace(quoted, "| 9.9999% |")
    problems = readme_drift(altered_text, generated_text)
    assert len(problems) == 1
    assert "9.9999%" in problems[0]
    committed = BASELINE_MARKDOWN_PATH.read_text(encoding="utf-8")
    assert readme_baseline_block(altered_text) != committed + "\n"


def readme_results_subsection(readme_text, section):
    """README text of results section '### N.' up to the next '### ' heading."""
    start = readme_text.index(f"\n### {section}. ")
    end = readme_text.find("\n### ", start + 1)
    if end == -1:
        end = readme_text.index("\n## ", start + 1)
    return readme_text[start:end]


def why_box_problems(readme_text):
    """'section N: ...' for each results section whose why box is not the shared text."""
    problems = []
    sections = sorted(SECTION_WHY_KEYS)
    for section in sections:
        block = section_why_block(section)
        subsection = readme_results_subsection(readme_text, section)
        count = subsection.count(block)
        if count != 1:
            problems.append(f"section {section}: shared why box found {count} times")
    return problems


def test_readme_why_boxes_are_the_shared_text():
    readme_text = README_PATH.read_text(encoding="utf-8")
    assert why_box_problems(readme_text) == []


def test_control_a_reworded_why_box_is_caught():
    readme_text = README_PATH.read_text(encoding="utf-8")
    original = section_why_block("2")
    altered_text = readme_text.replace(original, original.replace("too high", "too big"))
    assert altered_text != readme_text
    assert why_box_problems(altered_text) == ["section 2: shared why box found 0 times"]
