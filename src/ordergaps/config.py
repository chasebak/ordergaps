"""Store-hours configuration and estimator parameters."""

from dataclasses import dataclass

WEEKDAY_NAMES = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# Printed alongside every volume table this package produces.
CAVEAT = (
    "'volume' here means orders CREATED, not fulfilled: cancelled, voided and "
    "test orders consume order numbers and are counted."
)


def default_weekly_hours():
    """Example store schedule; minutes since midnight, None = closed."""
    hours = {
        0: None,  # Monday closed
        1: (660, 1200),  # Tuesday 11:00-20:00
        2: (660, 1200),  # Wednesday 11:00-20:00
        3: (660, 1200),  # Thursday 11:00-20:00
        4: (660, 1260),  # Friday 11:00-21:00
        5: (600, 1260),  # Saturday 10:00-21:00
        6: (660, 1080),  # Sunday 11:00-18:00
    }
    return hours


@dataclass(frozen=True)
class StoreHours:
    """Scheduled open hours per weekday (0=Monday .. 6=Sunday)."""

    weekly: dict

    @classmethod
    def default(cls):
        return cls(weekly=default_weekly_hours())

    def for_weekday(self, weekday):
        """(open_minute, close_minute) or None when closed that weekday."""
        return self.weekly.get(weekday)


@dataclass(frozen=True)
class EstimatorParams:
    # Minutes of fully-inside-session gap time a weekday needs before its
    # own learned rate is trusted; below this the pooled rate is used.
    min_rate_minutes: float = 240.0
    # A session end counts as "pinned" when the expected number of unseen
    # orders between it and the nearest seen order inside the session is at
    # most max(pin_tolerance_orders, pin_tolerance_fraction * expected day
    # total), so a full day's count is boundary-accurate to roughly that.
    pin_tolerance_orders: float = 5.0
    pin_tolerance_fraction: float = 0.05
    # An order-number drop larger than this (in time order) is treated as a
    # numbering reset rather than an out-of-order timestamp.
    reset_min_drop: int = 500
