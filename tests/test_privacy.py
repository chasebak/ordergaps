"""Public privacy check: no "N% -> M%" style share-growth line.

Why: the README once stated a real store-linked share change in that form;
it is cut until the store owner OKs it (plan Decision 14). This guards the
cut. Real store, vendor and staff names never appear here, not even in a
test: they belong in a private deny file kept outside the repo (planned).

Rule: fail on two WHOLE-number percentages joined by "->", "→", "to",
"up to" or a dash ("-", "–", "—") (e.g. a share "growing from N% to M%"
or "N%–M%"). A plain substring ban on "5%" is not used because simulated
results such as "0.0345%" contain it, and decimal percentages are left
alone because every simulated result in this repo is printed with
decimals. One exception: a DASH range whose both ends are vendor shares of
the simulated sweep (run_all.SHARES as whole percents: 2, 5, 10, 20, 35,
60) is allowed, because it describes the sweep, not a measured change.
Arrows and "to" get no exception: "from N% to M%" reads as a change even
when N and M are sweep values.
Failure mode: a real share written with decimals, joined by other words
("N% and M%", "N%, then M%"), or as a dash range of two sweep values would
slip through; the private deny check is the second layer.
"""

import re
from pathlib import Path

from ordergaps.evaluation.run_all import SHARES

REPO_ROOT = Path(__file__).resolve().parent.parent
# "->" is listed before "-" so an arrow is never read as a dash. The number
# after the joiner must end in "%" right away, so decimals ("0.5%") never
# match there either.
SHARE_CHANGE_PATTERN = re.compile(
    r"(?<![\d.])(\d{1,3})\s?%\s*(->|→|up\s+to|to|-|–|—)\s*(\d{1,3})\s?%"
)
DASH_JOINERS = ("-", "–", "—")


def sweep_percents():
    """Whole-percent vendor shares the simulator sweeps (0.02 -> 2, ...)."""
    percents = set()
    for share in SHARES:
        percent = round(share * 100)
        percents.add(percent)
    return percents


def share_change_hits(text):
    """Matched share-change strings in text, minus sweep-only dash ranges."""
    allowed_percents = sweep_percents()
    hits = []
    matches = SHARE_CHANGE_PATTERN.finditer(text)
    for match in matches:
        low = int(match.group(1))
        joiner = match.group(2)
        high = int(match.group(3))
        is_dash = joiner in DASH_JOINERS
        both_in_sweep = low in allowed_percents and high in allowed_percents
        if is_dash and both_in_sweep:
            continue
        hits.append(match.group(0))
    return hits


def public_text_files():
    paths = [REPO_ROOT / "README.md", REPO_ROOT / "MODEL.md"]
    source_paths = sorted((REPO_ROOT / "src" / "ordergaps").rglob("*.py"))
    test_paths = sorted((REPO_ROOT / "tests").rglob("*.py"))
    paths.extend(source_paths)
    paths.extend(test_paths)
    return paths


def test_pattern_catches_share_change_and_spares_results():
    # Built from pieces so this file does not itself contain the pattern.
    low = "3" + "%"
    high = "12" + "%"
    sweep_low = "2" + "%"
    sweep_high = "60" + "%"
    assert share_change_hits("grew from " + low + " to " + high + " in production")
    assert share_change_hits("share " + low + " → " + high)
    assert share_change_hits("share " + low + " -> " + high)
    assert share_change_hits("from " + low + " up to " + high)
    assert share_change_hits("share " + low + "–" + high)
    assert share_change_hits("share " + low + " - " + high)
    # Arrows/"to" between two sweep values still read as a change.
    assert share_change_hits("from " + sweep_low + " to " + sweep_high)
    assert not share_change_hits("swept " + sweep_low + "–" + sweep_high)
    assert not share_change_hits("0.3771% at 2% share down to 0.0150%")
    assert not share_change_hits("[+0.89%, +1.25%]")
    assert not share_change_hits("| 2% | -0.5% |")


def test_no_share_change_lines_in_public_files():
    hits = []
    paths = public_text_files()
    for path in paths:
        text = path.read_text(encoding="utf-8")
        matches = share_change_hits(text)
        for match in matches:
            hits.append(f"{path.relative_to(REPO_ROOT)}: {match}")
    assert hits == []
