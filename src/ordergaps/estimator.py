"""The estimator: observed vendor orders in, store-volume tables out.

Pipeline: era split -> gaps -> sessions -> learned weekday rates -> gap
attribution -> day classification -> daily and weekly tables -> dispersion
check. See config.CAVEAT: volume means orders CREATED, not fulfilled.
"""

from dataclasses import dataclass, field
from datetime import timedelta

import pandas as pd

from ordergaps.attribution import attribute_gap
from ordergaps.classify import CLOSED, FULL, PARTIAL, SPREAD, classify_days
from ordergaps.config import WEEKDAY_NAMES, EstimatorParams, StoreHours
from ordergaps.dispersion import dispersion_check
from ordergaps.gaps import build_gaps, split_eras
from ordergaps.rates import learn_weekday_rates
from ordergaps.sessions import build_sessions, total_session_minutes

REQUIRED_COLUMNS = ("order_number", "created_at", "channel")


@dataclass
class EstimateResult:
    daily: pd.DataFrame
    weekly: pd.DataFrame
    full_day_averages: pd.DataFrame
    rates: object  # WeekdayRates
    dispersion: object  # DispersionResult
    anomalies: list
    gaps: list = field(default_factory=list)
    day_labels: dict = field(default_factory=dict)
    sessions: dict = field(default_factory=dict)
    first_seen: pd.Timestamp = None
    last_seen: pd.Timestamp = None
    unseen_attributed: float = 0.0
    unseen_skipped: int = 0


def _validate(observed):
    for column in REQUIRED_COLUMNS:
        if column not in observed.columns:
            raise ValueError(f"observed orders need a '{column}' column")
    if len(observed) < 2:
        raise ValueError("need at least two observed orders to form a gap")


def estimate(observed, store_hours=None, params=None):
    """Estimate store order volume from one vendor's observed orders.

    observed: DataFrame with order_number (int), created_at (timestamp),
    channel ('register' or 'online'); optional items and total columns
    enable vendor basket stats.
    """
    if store_hours is None:
        store_hours = StoreHours.default()
    if params is None:
        params = EstimatorParams()
    _validate(observed)

    frame = observed.copy()
    frame["created_at"] = pd.to_datetime(frame["created_at"])
    frame["order_number"] = frame["order_number"].astype("int64")
    has_baskets = "items" in frame.columns and "total" in frame.columns

    eras, anomalies = split_eras(frame, params.reset_min_drop)
    all_gaps = []
    for era_rows in eras:
        era_gaps, era_anomalies = build_gaps(era_rows)
        all_gaps.extend(era_gaps)
        anomalies.extend(era_anomalies)

    first_seen = frame["created_at"].min()
    last_seen = frame["created_at"].max()
    first_day = first_seen.date()
    last_day = last_seen.date()

    register_frame = frame[frame["channel"] == "register"]
    register_times = list(register_frame["created_at"])
    sessions = build_sessions(register_times, store_hours, first_day, last_day)
    session_minutes = total_session_minutes(sessions)

    rates = learn_weekday_rates(
        all_gaps,
        sessions,
        params,
        fallback_count=len(register_times),
        fallback_minutes=session_minutes,
    )

    unseen_in_store = {}
    unseen_online = {}
    unseen_attributed = 0.0
    for gap in all_gaps:
        attribution = attribute_gap(gap, sessions, rates)
        in_store_items = attribution.in_store_by_day.items()
        for day, amount in in_store_items:
            unseen_in_store[day] = unseen_in_store.get(day, 0.0) + amount
            unseen_attributed += amount
        online_items = attribution.online_by_day.items()
        for day, amount in online_items:
            unseen_online[day] = unseen_online.get(day, 0.0) + amount
            unseen_attributed += amount

    unseen_skipped = 0
    for anomaly in anomalies:
        unseen_skipped += anomaly.get("unseen_skipped", 0)

    seen_register = {}
    seen_online = {}
    seen_times_by_day = {}
    vendor_revenue = {}
    vendor_items = {}
    row_tuples = list(frame.itertuples(index=False))
    for row in row_tuples:
        day = row.created_at.date()
        if day not in seen_times_by_day:
            seen_times_by_day[day] = []
        seen_times_by_day[day].append(row.created_at)
        if row.channel == "register":
            seen_register[day] = seen_register.get(day, 0) + 1
        else:
            seen_online[day] = seen_online.get(day, 0) + 1
        if has_baskets:
            vendor_revenue[day] = vendor_revenue.get(day, 0.0) + float(row.total)
            vendor_items[day] = vendor_items.get(day, 0) + int(row.items)

    day_labels = classify_days(sessions, seen_times_by_day, rates, params)

    # A numbering reset destroys the whole span from the previous era's last
    # observation (span_start) to the first post-reset one (at); every ISO
    # week that span overlaps is flagged, not just the week of the post-reset
    # row. Other anomalies carry no span_start and flag only the week of at.
    anomaly_weeks = set()
    one_day = timedelta(days=1)
    for anomaly in anomalies:
        at = anomaly.get("at")
        if at is None:
            continue
        span_start = anomaly.get("span_start")
        if span_start is None:
            span_start = at
        span_day = span_start.date()
        last_span_day = at.date()
        while span_day <= last_span_day:
            iso = span_day.isocalendar()
            anomaly_weeks.add((iso.year, iso.week))
            span_day = span_day + one_day

    daily_rows = []
    day = first_day
    while day <= last_day:
        label = day_labels.get(day, CLOSED)
        day_seen_register = seen_register.get(day, 0)
        day_seen_online = seen_online.get(day, 0)
        day_unseen_in_store = unseen_in_store.get(day, 0.0)
        day_unseen_online = unseen_online.get(day, 0.0)
        est_in_store = day_seen_register + day_unseen_in_store
        est_online = day_seen_online + day_unseen_online
        covered = first_day < day < last_day
        daily_rows.append(
            {
                "day": day,
                "weekday": WEEKDAY_NAMES[day.weekday()],
                "label": label,
                "covered": covered,
                "seen_register": day_seen_register,
                "seen_online": day_seen_online,
                "unseen_in_store": day_unseen_in_store,
                "unseen_online": day_unseen_online,
                "est_in_store": est_in_store,
                "est_online": est_online,
                "est_total": est_in_store + est_online,
                "vendor_orders": day_seen_register + day_seen_online,
                "vendor_revenue": vendor_revenue.get(day, 0.0) if has_baskets else float("nan"),
                "vendor_items": vendor_items.get(day, 0) if has_baskets else float("nan"),
            }
        )
        day = day + one_day
    daily = pd.DataFrame(daily_rows)

    weekly = _build_weekly(daily, anomaly_weeks, has_baskets)

    full_counts_by_weekday = {}
    full_rows = daily[daily["label"] == FULL]
    full_row_tuples = list(full_rows.itertuples(index=False))
    for row in full_row_tuples:
        weekday = row.day.weekday()
        if weekday not in full_counts_by_weekday:
            full_counts_by_weekday[weekday] = []
        full_counts_by_weekday[weekday].append(row.est_in_store)
    dispersion = dispersion_check(full_counts_by_weekday)

    average_rows = []
    weekday_values = range(7)
    for weekday in weekday_values:
        counts = full_counts_by_weekday.get(weekday, [])
        if counts:
            mean_in_store = sum(counts) / len(counts)
        else:
            mean_in_store = float("nan")
        average_rows.append(
            {
                "weekday": WEEKDAY_NAMES[weekday],
                "n_full_days": len(counts),
                "mean_in_store": mean_in_store,
            }
        )
    full_day_averages = pd.DataFrame(average_rows)

    return EstimateResult(
        daily=daily,
        weekly=weekly,
        full_day_averages=full_day_averages,
        rates=rates,
        dispersion=dispersion,
        anomalies=anomalies,
        gaps=all_gaps,
        day_labels=day_labels,
        sessions=sessions,
        first_seen=first_seen,
        last_seen=last_seen,
        unseen_attributed=unseen_attributed,
        unseen_skipped=unseen_skipped,
    )


def _build_weekly(daily, anomaly_weeks, has_baskets):
    week_data = {}
    daily_tuples = list(daily.itertuples(index=False))
    for row in daily_tuples:
        iso = row.day.isocalendar()
        key = (iso.year, iso.week)
        if key not in week_data:
            week_data[key] = {
                "iso_year": iso.year,
                "iso_week": iso.week,
                "week_start": row.day - timedelta(days=row.day.weekday()),
                "est_total": 0.0,
                "est_in_store": 0.0,
                "est_online": 0.0,
                "vendor_orders": 0,
                "vendor_register_orders": 0,
                "vendor_revenue": 0.0,
                "vendor_items": 0,
                "n_days": 0,
                "n_covered_days": 0,
                "n_full": 0,
                "n_partial": 0,
                "n_spread": 0,
                "n_closed": 0,
                "has_anomaly": key in anomaly_weeks,
            }
        week = week_data[key]
        week["est_total"] += row.est_total
        week["est_in_store"] += row.est_in_store
        week["est_online"] += row.est_online
        week["vendor_orders"] += row.vendor_orders
        week["vendor_register_orders"] += row.seen_register
        if has_baskets:
            week["vendor_revenue"] += row.vendor_revenue
            week["vendor_items"] += row.vendor_items
        week["n_days"] += 1
        if row.covered:
            week["n_covered_days"] += 1
        if row.label == FULL:
            week["n_full"] += 1
        elif row.label == PARTIAL:
            week["n_partial"] += 1
        elif row.label == SPREAD:
            week["n_spread"] += 1
        elif row.label == CLOSED:
            week["n_closed"] += 1

    weekly_rows = []
    week_items = sorted(week_data.items())
    for key, week in week_items:
        if week["est_in_store"] > 0.0:
            vendor_share = week["vendor_register_orders"] / week["est_in_store"]
        else:
            vendor_share = float("nan")
        if week["vendor_orders"] > 0 and has_baskets:
            items_per_order = week["vendor_items"] / week["vendor_orders"]
        else:
            items_per_order = float("nan")
        week["vendor_share_in_store"] = vendor_share
        week["vendor_items_per_order"] = items_per_order
        if not has_baskets:
            week["vendor_revenue"] = float("nan")
            week["vendor_items"] = float("nan")
        week["fully_covered"] = week["n_days"] == 7 and week["n_covered_days"] == 7
        weekly_rows.append(week)
    return pd.DataFrame(weekly_rows)
