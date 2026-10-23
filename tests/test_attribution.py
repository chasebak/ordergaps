from datetime import date

import pandas as pd
import pytest

from ordergaps.attribution import attribute_gap, split_span
from ordergaps.config import StoreHours
from ordergaps.gaps import Gap
from ordergaps.rates import WeekdayRates
from ordergaps.sessions import build_sessions

TUESDAY = date(2025, 3, 4)
WEDNESDAY = date(2025, 3, 5)


@pytest.fixture
def sessions():
    return build_sessions([], StoreHours.default(), TUESDAY, WEDNESDAY)


@pytest.fixture
def flat_rates():
    return WeekdayRates(per_weekday={}, pooled=0.5)


def test_split_span_open_and_closed_minutes(sessions):
    t1 = pd.Timestamp("2025-03-04 19:00")
    t2 = pd.Timestamp("2025-03-05 12:00")
    spans = split_span(t1, t2, sessions)
    assert len(spans) == 2
    tuesday_span = spans[0]
    wednesday_span = spans[1]
    assert tuesday_span.open_minutes == 60.0  # 19:00-20:00
    assert tuesday_span.closed_minutes == 240.0  # 20:00-24:00
    assert wednesday_span.open_minutes == 60.0  # 11:00-12:00
    assert wednesday_span.closed_minutes == 660.0  # 00:00-11:00


def test_gap_fully_inside_goes_all_in_store(sessions, flat_rates):
    gap = Gap(100, pd.Timestamp("2025-03-04 12:00"), 111, pd.Timestamp("2025-03-04 13:00"))
    attribution = attribute_gap(gap, sessions, flat_rates)
    assert attribution.kind == "inside"
    assert attribution.in_store_by_day == {TUESDAY: 10.0}
    assert attribution.online_by_day == {}


def test_gap_fully_outside_spreads_online_by_closed_minutes(sessions, flat_rates):
    gap = Gap(200, pd.Timestamp("2025-03-04 21:00"), 209, pd.Timestamp("2025-03-05 01:00"))
    attribution = attribute_gap(gap, sessions, flat_rates)
    assert attribution.kind == "outside"
    assert attribution.in_store_by_day == {}
    assert attribution.online_by_day[TUESDAY] == pytest.approx(8 * 180.0 / 240.0)
    assert attribution.online_by_day[WEDNESDAY] == pytest.approx(8 * 60.0 / 240.0)


def test_mixed_gap_caps_in_store_at_expected(sessions, flat_rates):
    gap = Gap(1000, pd.Timestamp("2025-03-04 19:00"), 1101, pd.Timestamp("2025-03-05 12:00"))
    attribution = attribute_gap(gap, sessions, flat_rates)
    assert attribution.kind == "mixed"
    # expected in-store = (60 + 60) minutes * 0.5/minute = 60 orders
    assert attribution.in_store_total == pytest.approx(60.0)
    assert attribution.online_total == pytest.approx(40.0)
    assert attribution.in_store_by_day[TUESDAY] == pytest.approx(30.0)
    assert attribution.in_store_by_day[WEDNESDAY] == pytest.approx(30.0)
    # online spread by closed minutes: 240 on Tuesday night, 660 Wednesday morning
    assert attribution.online_by_day[TUESDAY] == pytest.approx(40.0 * 240.0 / 900.0)
    assert attribution.online_by_day[WEDNESDAY] == pytest.approx(40.0 * 660.0 / 900.0)


def test_mixed_gap_smaller_than_expected_stays_in_store(sessions, flat_rates):
    gap = Gap(1000, pd.Timestamp("2025-03-04 19:00"), 1011, pd.Timestamp("2025-03-05 12:00"))
    attribution = attribute_gap(gap, sessions, flat_rates)
    assert attribution.in_store_total == pytest.approx(10.0)
    assert attribution.online_total == pytest.approx(0.0)


def test_attribution_conserves_unseen(sessions, flat_rates):
    gap = Gap(500, pd.Timestamp("2025-03-04 10:00"), 578, pd.Timestamp("2025-03-05 23:00"))
    attribution = attribute_gap(gap, sessions, flat_rates)
    total = attribution.in_store_total + attribution.online_total
    assert total == pytest.approx(gap.unseen)


def test_zero_length_span_inside_session(sessions, flat_rates):
    ts = pd.Timestamp("2025-03-04 12:00")
    gap = Gap(300, ts, 305, ts)
    attribution = attribute_gap(gap, sessions, flat_rates)
    assert attribution.in_store_by_day == {TUESDAY: 4.0}


def test_empty_gap_attributes_nothing(sessions, flat_rates):
    gap = Gap(400, pd.Timestamp("2025-03-04 12:00"), 401, pd.Timestamp("2025-03-04 12:05"))
    attribution = attribute_gap(gap, sessions, flat_rates)
    assert attribution.kind == "empty"
    assert attribution.in_store_by_day == {}
    assert attribution.online_by_day == {}
