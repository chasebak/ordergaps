from datetime import date

import pandas as pd

from ordergaps.config import StoreHours
from ordergaps.sessions import build_sessions, total_session_minutes

TUESDAY = date(2025, 3, 4)
MONDAY = date(2025, 3, 3)


def test_scheduled_session_without_stretch():
    sessions = build_sessions([], StoreHours.default(), TUESDAY, TUESDAY)
    session = sessions[TUESDAY]
    assert session is not None
    assert session.start == pd.Timestamp("2025-03-04 11:00")
    assert session.end == pd.Timestamp("2025-03-04 20:00")
    assert session.scheduled
    assert not session.stretched_start
    assert not session.stretched_end


def test_session_stretches_to_early_register_order():
    register_times = [pd.Timestamp("2025-03-04 09:15")]
    sessions = build_sessions(register_times, StoreHours.default(), TUESDAY, TUESDAY)
    session = sessions[TUESDAY]
    assert session.start == pd.Timestamp("2025-03-04 09:15")
    assert session.stretched_start
    assert session.end == pd.Timestamp("2025-03-04 20:00")


def test_session_stretches_to_late_register_order():
    register_times = [pd.Timestamp("2025-03-04 21:40")]
    sessions = build_sessions(register_times, StoreHours.default(), TUESDAY, TUESDAY)
    session = sessions[TUESDAY]
    assert session.end == pd.Timestamp("2025-03-04 21:40")
    assert session.stretched_end


def test_closed_day_with_register_orders_counts_as_open():
    register_times = [
        pd.Timestamp("2025-03-03 12:00"),
        pd.Timestamp("2025-03-03 15:30"),
    ]
    sessions = build_sessions(register_times, StoreHours.default(), MONDAY, MONDAY)
    session = sessions[MONDAY]
    assert session is not None
    assert not session.scheduled
    assert session.start == pd.Timestamp("2025-03-03 12:00")
    assert session.end == pd.Timestamp("2025-03-03 15:30")


def test_closed_day_without_register_orders_stays_closed():
    sessions = build_sessions([], StoreHours.default(), MONDAY, MONDAY)
    assert sessions[MONDAY] is None


def test_online_orders_do_not_stretch_sessions():
    # build_sessions receives register times only; the estimator filters.
    sessions = build_sessions([], StoreHours.default(), MONDAY, TUESDAY)
    assert sessions[MONDAY] is None
    assert total_session_minutes(sessions) == 9 * 60.0
