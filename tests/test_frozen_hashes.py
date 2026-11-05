"""Frozen-number guards: the simulator's random draws and the README's
committed results must not move when forecasting code is added.

Why: every README number in "Results" sections 1-3 comes from simulate()
on the frozen seeds. Adding simulator.true_daily_mean (a read-only helper)
must not change a single random draw, so the hashes below were pinned from
the code BEFORE that helper was added (2026-11-04), and they must still
match after it.

Rule: a hash is sha256 over a canonical byte serialization (int64 arrays of
timestamps in ns, order numbers, channel codes and flags; basket items and
totals; little-endian), so it does not depend on pandas' own hashing or CSV
formatting. Only the first 16 hex characters are pinned (a full 64-hex
string would trip the public privacy scan for hash-shaped strings).

The README hash excludes sections 1.1, 1.2 and 1.3 (the uniform baseline,
the full-day yield and the estimation KPI table, added 2026-11-04 AFTER the
pin): they are cut out
before hashing, so every byte that existed at pin time must still be there
unchanged, while the added block is checked on its own against
results/baseline_yield.md (tests/test_readme_drift.py).

The why boxes under tables 1, 2 and 3 (added 2026-11-04 with the Gap
Explorer page, which shares their text) are cut out the same way: each is
removed as the exact block run_all.section_why_block() prints plus its blank
line, so a why box that is reworded on one side no longer cuts cleanly and
moves the hash; tests/test_readme_drift.py checks the boxes themselves.

Failure mode guarded: a simulator edit that adds, removes or reorders an
rng call (silently shifting every frozen number), and a hand edit of the
README's frozen tables. The control test shows the hash does move when one
draw moves.
"""

import hashlib
from pathlib import Path

import numpy as np

from ordergaps.evaluation.run_all import (
    SECTION_WHY_KEYS,
    frozen_base_config,
    section_why_block,
)
from ordergaps.simulator import SimConfig, simulate

README_PATH = Path(__file__).resolve().parent.parent / "README.md"

# Pinned before simulator.true_daily_mean existed.
PINNED_SMALL_SIM = "3d95a6e17b44f6aa"
PINNED_FROZEN_SIM_101 = "4e043ff085cc6fcb"
# README "## Results" text through the end of section 3, byte for byte,
# with sections 1.1 / 1.2 cut out (they did not exist at pin time).
PINNED_README_FROZEN_RESULTS = "f4545dccd8693f82"
HASH_PREFIX_LENGTH = 16
ADDED_BLOCK_HEADING = "#### 1.1 "
ADDED_BLOCK_END = "### 2. "


def _int64_bytes(values):
    array = np.asarray(values, dtype="<i8")
    return array.tobytes()


def _float64_bytes(values):
    array = np.asarray(values, dtype="<f8")
    return array.tobytes()


def sim_digest(sim):
    """First 16 hex chars of sha256 over the canonical bytes of one run."""
    orders = sim.orders
    observed = sim.observed
    created_ns = orders["created_at"].astype("datetime64[ns]").astype("int64")
    channel_codes = (orders["channel"] == "register").astype("int64")
    digest = hashlib.sha256()
    digest.update(_int64_bytes(created_ns))
    digest.update(_int64_bytes(orders["order_number"]))
    digest.update(_int64_bytes(channel_codes))
    digest.update(_int64_bytes(orders["is_void"].astype("int64")))
    digest.update(_int64_bytes(orders["is_vendor"].astype("int64")))
    digest.update(_int64_bytes(observed["order_number"]))
    digest.update(_int64_bytes(observed["items"]))
    digest.update(_float64_bytes(observed["total"]))
    return digest.hexdigest()[:HASH_PREFIX_LENGTH]


def readme_frozen_results_block(readme_text):
    """README text from '## Results' up to section 4 (or the next '## ')."""
    start = readme_text.index("## Results")
    rest = readme_text[start:]
    end_candidates = []
    section_four = rest.find("\n### 4.")
    if section_four != -1:
        end_candidates.append(section_four + 1)
    next_top = rest.find("\n## ", 1)
    if next_top != -1:
        end_candidates.append(next_top + 1)
    end = min(end_candidates)
    return rest[:end]


def without_added_block(block):
    """The block minus the added sections 1.1 / 1.2 (from the 1.1 heading up
    to the section-2 heading); unchanged when they are absent."""
    start = block.find(ADDED_BLOCK_HEADING)
    if start == -1:
        return block
    end = block.index(ADDED_BLOCK_END, start)
    return block[:start] + block[end:]


def without_why_boxes(block):
    """The block minus the why boxes of sections 1-3 (each removed as its
    exact generated text plus the blank line after it)."""
    trimmed = block
    sections = sorted(SECTION_WHY_KEYS)
    for section in sections:
        why_block = section_why_block(section) + "\n\n"
        trimmed = trimmed.replace(why_block, "", 1)
    return trimmed


def pinned_part_of(readme_text):
    block = readme_frozen_results_block(readme_text)
    return without_why_boxes(without_added_block(block))


def text_digest(text):
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return digest[:HASH_PREFIX_LENGTH]


def test_small_simulation_draws_are_frozen():
    sim = simulate(SimConfig(n_weeks=2), seed=5)
    assert sim_digest(sim) == PINNED_SMALL_SIM


def test_frozen_config_draws_are_frozen():
    sim = simulate(frozen_base_config(), seed=101)
    assert sim_digest(sim) == PINNED_FROZEN_SIM_101


def test_readme_frozen_results_are_byte_identical():
    readme_text = README_PATH.read_text(encoding="utf-8")
    block = readme_frozen_results_block(readme_text)
    assert block.startswith("## Results")
    assert "13713/13713" in block
    pinned_part = pinned_part_of(readme_text)
    assert "#### 1." not in pinned_part
    assert "Why this metric" not in pinned_part
    assert text_digest(pinned_part) == PINNED_README_FROZEN_RESULTS


def test_control_cutting_the_added_block_cannot_hide_a_frozen_edit():
    # Control that must fail: one frozen section-1 number edited next to the
    # added block still moves the hash after the cut.
    readme_text = README_PATH.read_text(encoding="utf-8")
    altered_text = readme_text.replace("| 0.02 | 0.020 | 0.3771% |", "| 0.02 | 0.020 | 0.3772% |")
    assert altered_text != readme_text
    assert text_digest(pinned_part_of(altered_text)) != PINNED_README_FROZEN_RESULTS


def test_control_a_reworded_why_box_moves_the_hash():
    # Control that must fail: a why box edited in the README only no longer
    # matches the generated block, so it is not cut and the hash moves.
    readme_text = README_PATH.read_text(encoding="utf-8")
    original = section_why_block("3")
    assert readme_text.count(original) == 1
    reworded = original.replace("Which of these numbers", "Which numbers")
    altered_text = readme_text.replace(original, reworded)
    assert text_digest(pinned_part_of(altered_text)) != PINNED_README_FROZEN_RESULTS


def test_control_digest_moves_when_one_draw_moves():
    sim = simulate(SimConfig(n_weeks=2), seed=5)
    before = sim_digest(sim)
    sim.observed.loc[0, "items"] = int(sim.observed.loc[0, "items"]) + 1
    after = sim_digest(sim)
    assert after != before
