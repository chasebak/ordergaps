"""README-number drift test: every result number the README quotes must be
exactly what the generator prints.

Why: the README's "Results" tables are copied by hand from the output of
`python -m ordergaps.evaluation.run_all` (out/results.md). out/ is not
committed, so the source of truth in CI is the generator itself:
run_all.results_markdown() fed by the frozen matrix. If the code, the seeds
or the README change without the other, this test fails.

Rule (what is checked):
- Only the README section "## Results ..." up to the next "## " heading.
- Inside it, only the lines the generator emits:
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

import re
from pathlib import Path

from ordergaps.evaluation.experiments import (
    acceptance_check,
    counted_vs_allocated,
    error_curve,
    length_bias_summary,
)
from ordergaps.evaluation.run_all import TABLE_SHARE, results_markdown

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
NUMBER_PATTERN = re.compile(r"[+-]?\d+(?:\.\d+)?%?")
SECTION_PATTERN = re.compile(r"^#{2,3}\s+(\d+)\.")
VERDICT_PATTERN = re.compile(r"Acceptance band:.*?(PASS|FAIL)")
# 6 + 6 + 5 data rows plus 3 header rows exist today; fewer means the parser
# stopped seeing the tables and the test would pass vacuously.
MIN_CHECKED_ROWS = 20


def generated_results_text(records):
    """The exact text run_all writes to out/results.md, minus timing."""
    curve = error_curve(records)
    check = acceptance_check(curve)
    bias_summary = length_bias_summary(records)
    table = counted_vs_allocated(records, TABLE_SHARE)
    text = results_markdown(curve, check, bias_summary, table, elapsed_seconds=0.0)
    return text


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
            section = int(heading_match.group(1))
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
