"""Daily sessions: scheduled store hours stretched by seen register orders."""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd


@dataclass
class Session:
    day: object  # datetime.date
    start: pd.Timestamp
    end: pd.Timestamp
    scheduled: bool  # True when the store-hours config lists the day as open
    stretched_start: bool
    stretched_end: bool

    def minutes(self):
        return (self.end - self.start).total_seconds() / 60.0


def group_times_by_day(times):
    by_day = {}
    for ts in times:
        day = ts.date()
        if day not in by_day:
            by_day[day] = []
        by_day[day].append(ts)
    return by_day


def build_sessions(register_times, store_hours, first_day, last_day):
    """Map each calendar day in [first_day, last_day] to a Session or None.

    A day's scheduled hours are stretched to its first/last seen register
    order; a scheduled-closed day with seen register orders counts as open
    (from its first to its last register order).
    """
    register_by_day = group_times_by_day(register_times)
    sessions = {}
    one_day = timedelta(days=1)
    day = first_day
    while day <= last_day:
        scheduled = store_hours.for_weekday(day.weekday())
        day_registers = register_by_day.get(day, [])
        session = None
        if scheduled is not None:
            open_minute, close_minute = scheduled
            start = pd.Timestamp(day) + pd.Timedelta(minutes=open_minute)
            end = pd.Timestamp(day) + pd.Timedelta(minutes=close_minute)
            stretched_start = False
            stretched_end = False
            if day_registers:
                first_register = min(day_registers)
                last_register = max(day_registers)
                if first_register < start:
                    start = first_register
                    stretched_start = True
                if last_register > end:
                    end = last_register
                    stretched_end = True
            session = Session(day, start, end, True, stretched_start, stretched_end)
        elif day_registers:
            first_register = min(day_registers)
            last_register = max(day_registers)
            session = Session(day, first_register, last_register, False, True, True)
        sessions[day] = session
        day = day + one_day
    return sessions


def total_session_minutes(sessions):
    total = 0.0
    session_values = sessions.values()
    for session in session_values:
        if session is not None:
            total += session.minutes()
    return total
