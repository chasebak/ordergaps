"""Ground-truth simulator: the thing the production system never had.

Seeded nonhomogeneous Poisson arrivals (weekday x hour-of-day), store hours
with closed days, two channels (in-store only during open hours; online any
time), and a focal vendor whose orders are the only observations. Planted
pathologies: cancelled/void orders that consume numbers, a numbering reset,
and rate drift over weeks.

For a fixed seed the arrival pattern is identical across vendor shares
(vendor membership is thresholded on one shared uniform draw per order), so
share sweeps are paired comparisons.
"""

import math
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from ordergaps.config import StoreHours


@dataclass(frozen=True)
class SimConfig:
    n_weeks: int = 10
    start_day: str = "2025-03-03"  # a Monday; pinned for reproducibility
    store_hours: StoreHours = field(default_factory=StoreHours.default)
    # Expected in-store orders per day, Monday..Sunday (0 where closed).
    in_store_daily: tuple = (0.0, 150.0, 150.0, 160.0, 200.0, 280.0, 130.0)
    online_daily: float = 90.0  # expected online orders per 24 h day
    vendor_share: float = 0.10
    void_rate: float = 0.02
    reset_week: int | None = None  # numbering restarts at the start of this week (0-based)
    reset_new_base: int = 101
    start_base: int = 1001
    drift_per_week: float = 0.0  # multiplicative rate drift per week
    basket_extra_items_mean: float = 1.2
    price_log_mean: float = 3.0
    price_log_sigma: float = 0.6


def with_share(config, share):
    return replace(config, vendor_share=share)


def _in_store_hour_weight(hour_center):
    midday = math.exp(-((hour_center - 13.0) ** 2) / 18.0)
    evening = math.exp(-((hour_center - 17.5) ** 2) / 10.0)
    return 0.35 + 0.9 * midday + 1.1 * evening


def _online_hour_weight(hour_center):
    evening = math.exp(-((hour_center - 21.0) ** 2) / 20.0)
    return 0.5 + 1.0 * evening


@dataclass
class SimResult:
    config: SimConfig
    seed: int
    orders: pd.DataFrame  # every created order, voids included
    observed: pd.DataFrame  # what the vendor sees: its own non-void orders
    truth_daily: pd.DataFrame
    truth_weekly: pd.DataFrame
    true_weekday_rates: dict  # weekday -> orders per scheduled open minute
    fraction_observed: float
    reset_time: pd.Timestamp | None


def simulate(config, seed):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp(config.start_day)
    n_days = config.n_weeks * 7
    hour_values = range(24)

    times = []
    channels = []
    day_index_values = range(n_days)
    for day_index in day_index_values:
        day_ts = start + pd.Timedelta(days=day_index)
        weekday = day_ts.weekday()
        week_index = day_index // 7
        drift = 1.0 + config.drift_per_week * week_index

        scheduled = config.store_hours.for_weekday(weekday)
        if scheduled is not None and config.in_store_daily[weekday] > 0.0:
            open_minute, close_minute = scheduled
            window_weights = []
            windows = []
            for hour in hour_values:
                hour_begin = hour * 60.0
                hour_end = hour_begin + 60.0
                overlap_begin = max(hour_begin, float(open_minute))
                overlap_end = min(hour_end, float(close_minute))
                if overlap_end <= overlap_begin:
                    continue
                overlap_fraction = (overlap_end - overlap_begin) / 60.0
                weight = _in_store_hour_weight(hour + 0.5) * overlap_fraction
                windows.append((overlap_begin, overlap_end))
                window_weights.append(weight)
            weight_total = sum(window_weights)
            daily_mean = config.in_store_daily[weekday] * drift
            window_values = range(len(windows))
            for window_index in window_values:
                begin_minute, end_minute = windows[window_index]
                lam = daily_mean * window_weights[window_index] / weight_total
                count = int(rng.poisson(lam))
                if count == 0:
                    continue
                uniforms = rng.random(count)
                for u in uniforms:
                    minute = begin_minute + u * (end_minute - begin_minute)
                    times.append(day_ts + pd.Timedelta(minutes=minute))
                    channels.append("register")

        online_weights = []
        for hour in hour_values:
            online_weights.append(_online_hour_weight(hour + 0.5))
        online_weight_total = sum(online_weights)
        online_mean = config.online_daily * drift
        for hour in hour_values:
            lam = online_mean * online_weights[hour] / online_weight_total
            count = int(rng.poisson(lam))
            if count == 0:
                continue
            uniforms = rng.random(count)
            for u in uniforms:
                minute = hour * 60.0 + u * 60.0
                times.append(day_ts + pd.Timedelta(minutes=minute))
                channels.append("online")

    orders = pd.DataFrame({"created_at": times, "channel": channels})
    orders = orders.sort_values("created_at", kind="stable").reset_index(drop=True)
    n_orders = len(orders)

    reset_time = None
    if config.reset_week is not None:
        reset_time = start + pd.Timedelta(days=7 * config.reset_week)
    numbers = []
    counter = config.start_base
    reset_done = False
    created_values = list(orders["created_at"])
    for created_at in created_values:
        if reset_time is not None and not reset_done and created_at >= reset_time:
            counter = config.reset_new_base
            reset_done = True
        numbers.append(counter)
        counter += 1
    orders["order_number"] = numbers

    # One shared uniform per order: void first, vendor membership second, so
    # the same seed gives nested vendor sets as the share grows.
    void_uniforms = rng.random(n_orders)
    vendor_uniforms = rng.random(n_orders)
    orders["is_void"] = void_uniforms < config.void_rate
    orders["is_vendor"] = vendor_uniforms < config.vendor_share

    observed_mask = orders["is_vendor"] & ~orders["is_void"]
    observed = orders[observed_mask].copy().reset_index(drop=True)
    n_observed = len(observed)
    extra_items = rng.poisson(config.basket_extra_items_mean, size=n_observed)
    item_counts = []
    totals = []
    observed_values = range(n_observed)
    for index in observed_values:
        item_count = 1 + int(extra_items[index])
        prices = rng.lognormal(config.price_log_mean, config.price_log_sigma, size=item_count)
        item_counts.append(item_count)
        totals.append(round(float(np.sum(prices)), 2))
    observed["items"] = item_counts
    observed["total"] = totals
    observed = observed[["order_number", "created_at", "channel", "items", "total"]]

    truth_daily = _truth_daily(orders, start, n_days)
    truth_weekly = _truth_weekly(truth_daily)
    true_weekday_rates = _true_weekday_rates(orders, config, start, n_days)
    fraction_observed = n_observed / n_orders if n_orders else 0.0

    return SimResult(
        config=config,
        seed=seed,
        orders=orders,
        observed=observed,
        truth_daily=truth_daily,
        truth_weekly=truth_weekly,
        true_weekday_rates=true_weekday_rates,
        fraction_observed=fraction_observed,
        reset_time=reset_time,
    )


def _truth_daily(orders, start, n_days):
    in_store_by_day = {}
    online_by_day = {}
    order_tuples = list(orders.itertuples(index=False))
    for row in order_tuples:
        day = row.created_at.date()
        if row.channel == "register":
            in_store_by_day[day] = in_store_by_day.get(day, 0) + 1
        else:
            online_by_day[day] = online_by_day.get(day, 0) + 1
    rows = []
    day_index_values = range(n_days)
    for day_index in day_index_values:
        day = (start + pd.Timedelta(days=day_index)).date()
        in_store = in_store_by_day.get(day, 0)
        online = online_by_day.get(day, 0)
        rows.append(
            {"day": day, "in_store": in_store, "online": online, "total": in_store + online}
        )
    return pd.DataFrame(rows)


def _truth_weekly(truth_daily):
    week_data = {}
    daily_tuples = list(truth_daily.itertuples(index=False))
    for row in daily_tuples:
        iso = row.day.isocalendar()
        key = (iso.year, iso.week)
        if key not in week_data:
            week_data[key] = {
                "iso_year": iso.year,
                "iso_week": iso.week,
                "in_store": 0,
                "online": 0,
                "total": 0,
            }
        week = week_data[key]
        week["in_store"] += row.in_store
        week["online"] += row.online
        week["total"] += row.total
    rows = []
    week_items = sorted(week_data.items())
    for key, week in week_items:
        rows.append(week)
    return pd.DataFrame(rows)


def _true_weekday_rates(orders, config, start, n_days):
    """Orders of ANY channel created inside scheduled hours, per open minute."""
    counts = {}
    minutes = {}
    day_index_values = range(n_days)
    for day_index in day_index_values:
        day_ts = start + pd.Timedelta(days=day_index)
        weekday = day_ts.weekday()
        scheduled = config.store_hours.for_weekday(weekday)
        if scheduled is None:
            continue
        open_minute, close_minute = scheduled
        minutes[weekday] = minutes.get(weekday, 0.0) + (close_minute - open_minute)
    order_tuples = list(orders.itertuples(index=False))
    for row in order_tuples:
        weekday = row.created_at.weekday()
        scheduled = config.store_hours.for_weekday(weekday)
        if scheduled is None:
            continue
        open_minute, close_minute = scheduled
        minute_of_day = row.created_at.hour * 60.0 + row.created_at.minute + (
            row.created_at.second / 60.0
        )
        if open_minute <= minute_of_day <= close_minute:
            counts[weekday] = counts.get(weekday, 0) + 1
    rates = {}
    weekday_values = range(7)
    for weekday in weekday_values:
        weekday_minutes = minutes.get(weekday, 0.0)
        if weekday_minutes > 0.0:
            rates[weekday] = counts.get(weekday, 0) / weekday_minutes
    return rates


def true_daily_mean(config, day_index, channel="total"):
    """The TRUE expected number of orders created on one simulated day.

    Read-only: computed from the config alone, with no random draws, so
    adding it changes nothing simulate() produces (tests/test_frozen_hashes.py
    pins simulate() output from before this helper existed).

    Rule (mirrors simulate()): with drift = 1 + drift_per_week * (day_index // 7),
    in-store mean = in_store_daily[weekday] * drift on a scheduled-open day
    (0 when the day is closed or its in-store rate is 0), online mean =
    online_daily * drift; the hour-window weights inside simulate() are
    normalized, so they sum back to these daily means exactly. channel is
    "in_store", "online" or "total" (voids included: they are created orders).

    This is the forecast oracle's input: the best any model can do, because
    the realized count is Poisson around it. Failure mode: it goes stale if
    simulate()'s rate logic changes without it; tests/test_forecast.py checks
    parity (simulated daily means match it within Poisson error).
    """
    day_ts = pd.Timestamp(config.start_day) + pd.Timedelta(days=day_index)
    weekday = day_ts.weekday()
    week_index = day_index // 7
    drift = 1.0 + config.drift_per_week * week_index
    in_store_mean = 0.0
    scheduled = config.store_hours.for_weekday(weekday)
    if scheduled is not None and config.in_store_daily[weekday] > 0.0:
        in_store_mean = config.in_store_daily[weekday] * drift
    online_mean = config.online_daily * drift
    if channel == "in_store":
        return in_store_mean
    if channel == "online":
        return online_mean
    if channel == "total":
        return in_store_mean + online_mean
    raise ValueError(f"unknown channel {channel!r}")
