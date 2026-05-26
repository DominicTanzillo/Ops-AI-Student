"""US Federal holiday calendar lookup used by the data quality validator.

Kept separate from check_data_quality.py so the calendar is editable independently
of the validation logic (e.g., adding observed-Monday rules, regional holidays, etc.).
"""
from datetime import date as _date
from typing import Iterable, Union

# US Federal holidays for years that may appear in this dataset (2023 - 2026).
# Each entry maps date -> human-readable name. Add years/observed-rules as needed.
US_FEDERAL_HOLIDAYS = {
    # 2023
    _date(2023, 1, 1):  "New Year's Day",
    _date(2023, 1, 16): "MLK Day",
    _date(2023, 2, 20): "Presidents Day",
    _date(2023, 5, 29): "Memorial Day",
    _date(2023, 6, 19): "Juneteenth",
    _date(2023, 7, 4):  "Independence Day",
    _date(2023, 9, 4):  "Labor Day",
    _date(2023, 10, 9): "Columbus Day",
    _date(2023, 11, 11): "Veterans Day",
    _date(2023, 11, 23): "Thanksgiving",
    _date(2023, 12, 25): "Christmas Day",
    # 2024
    _date(2024, 1, 1):  "New Year's Day",
    _date(2024, 1, 15): "MLK Day",
    _date(2024, 2, 19): "Presidents Day",
    _date(2024, 5, 27): "Memorial Day",
    _date(2024, 6, 19): "Juneteenth",
    _date(2024, 7, 4):  "Independence Day",
    _date(2024, 9, 2):  "Labor Day",
    _date(2024, 10, 14): "Columbus Day",
    _date(2024, 11, 11): "Veterans Day",
    _date(2024, 11, 28): "Thanksgiving",
    _date(2024, 12, 25): "Christmas Day",
    # 2025
    _date(2025, 1, 1):  "New Year's Day",
    _date(2025, 1, 20): "MLK Day",
    _date(2025, 2, 17): "Presidents Day",
    _date(2025, 5, 26): "Memorial Day",
    _date(2025, 6, 19): "Juneteenth",
    _date(2025, 7, 4):  "Independence Day",
    _date(2025, 9, 1):  "Labor Day",
    _date(2025, 10, 13): "Columbus Day",
    _date(2025, 11, 11): "Veterans Day",
    _date(2025, 11, 27): "Thanksgiving",
    _date(2025, 12, 25): "Christmas Day",
    # 2026
    _date(2026, 1, 1):  "New Year's Day",
    _date(2026, 1, 19): "MLK Day",
    _date(2026, 2, 16): "Presidents Day",
    _date(2026, 5, 25): "Memorial Day",
    _date(2026, 6, 19): "Juneteenth",
    _date(2026, 7, 3):  "Independence Day (observed)",
    _date(2026, 7, 4):  "Independence Day",
    _date(2026, 9, 7):  "Labor Day",
    _date(2026, 10, 12): "Columbus Day",
    _date(2026, 11, 11): "Veterans Day",
    _date(2026, 11, 26): "Thanksgiving",
    _date(2026, 12, 25): "Christmas Day",
}


def is_real_holiday(d: Union[_date, str]) -> bool:
    """Return True if the given date is a known US Federal holiday in the calendar."""
    if isinstance(d, str):
        d = _date.fromisoformat(d)
    return d in US_FEDERAL_HOLIDAYS


def holiday_name(d: Union[_date, str]) -> str:
    """Return the holiday name for a date, or 'regular' if not a holiday."""
    if isinstance(d, str):
        d = _date.fromisoformat(d)
    return US_FEDERAL_HOLIDAYS.get(d, "regular")


def real_holiday_dates(year: int) -> Iterable[_date]:
    """All known holiday dates for a given year."""
    return [d for d in US_FEDERAL_HOLIDAYS if d.year == year]
