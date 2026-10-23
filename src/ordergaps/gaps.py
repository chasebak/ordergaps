"""Order-sequence handling: era splitting, gap construction, pathologies.

The arithmetic core: for consecutive seen order numbers n1 < n2, exactly
n2 - n1 - 1 store orders were created in between. That count is exact;
everything else in this package is about WHEN and through WHICH channel.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass
class Gap:
    n1: int
    t1: pd.Timestamp
    n2: int
    t2: pd.Timestamp

    @property
    def unseen(self):
        return self.n2 - self.n1 - 1


def split_eras(observed, reset_min_drop):
    """Split time-ordered observed orders into numbering eras.

    A drop of more than reset_min_drop in order number while time moves
    forward is treated as a numbering reset and starts a new era. Small
    backward steps are left alone (those are out-of-order timestamps and
    are handled per gap in build_gaps).

    A reset anomaly records span_start (the previous era's last created_at)
    alongside at (the first post-reset created_at): the destroyed cross-reset
    span runs between the two, and often lies mostly BEFORE the reset.

    Returns (eras, anomalies); each era is a list of row namedtuples.
    """
    ordered = observed.sort_values("created_at", kind="stable")
    row_tuples = list(ordered.itertuples(index=False))
    eras = []
    anomalies = []
    current_rows = []
    previous_number = None
    previous_created_at = None
    for row in row_tuples:
        number = int(row.order_number)
        if previous_number is not None and number < previous_number - reset_min_drop:
            anomalies.append(
                {
                    "kind": "numbering_reset",
                    "at": row.created_at,
                    "span_start": previous_created_at,
                    "detail": f"order number dropped {previous_number} -> {number}",
                    "unseen_skipped": 0,
                }
            )
            eras.append(current_rows)
            current_rows = []
        current_rows.append(row)
        previous_number = number
        previous_created_at = row.created_at
    if current_rows:
        eras.append(current_rows)
    return eras, anomalies


def _order_number_of(row):
    return int(row.order_number)


def build_gaps(era_rows):
    """Consecutive-number gaps within one era, with pathology handling.

    Out-of-order rows (timestamp regression between consecutive numbers)
    and duplicate numbers produce an anomaly record and the gap is skipped;
    its unseen count is reported, never attributed.
    """
    by_number = sorted(era_rows, key=_order_number_of)
    gaps = []
    anomalies = []
    previous = None
    for row in by_number:
        if previous is None:
            previous = row
            continue
        n1 = int(previous.order_number)
        n2 = int(row.order_number)
        if n2 == n1:
            anomalies.append(
                {
                    "kind": "duplicate_number",
                    "at": row.created_at,
                    "detail": f"order number {n2} seen twice",
                    "unseen_skipped": 0,
                }
            )
            previous = row
            continue
        if row.created_at < previous.created_at:
            unseen_skipped = n2 - n1 - 1
            anomalies.append(
                {
                    "kind": "out_of_order_timestamp",
                    "at": row.created_at,
                    "detail": f"timestamp regressed between orders {n1} and {n2}",
                    "unseen_skipped": unseen_skipped,
                }
            )
            previous = row
            continue
        gaps.append(Gap(n1, previous.created_at, n2, row.created_at))
        previous = row
    return gaps, anomalies
