"""
Centralized Datetime Utilities for Indian Standard Time (IST, UTC+5:30).
Ensures consistent Indian Standard Time formatting, parsing, and serialization
across backend services, database records, reports, and emails.
"""
from datetime import datetime, timezone, timedelta
from typing import Optional

# Indian Standard Time: UTC+5:30
IST = timezone(timedelta(hours=5, minutes=30), name="IST")

def get_ist_now() -> datetime:
    """Return current timezone-aware datetime in Indian Standard Time (IST, UTC+5:30)."""
    return datetime.now(IST)

def get_ist_iso() -> str:
    """Return current IST timestamp formatted as ISO 8601 string (e.g. 2026-09-24T00:15:30.123456+05:30)."""
    return datetime.now(IST).isoformat()

def get_ist_filename_timestamp() -> str:
    """Return compact numeric timestamp in IST for files (e.g. 20260924_001530)."""
    return datetime.now(IST).strftime("%Y%m%d_%H%M%S")

def format_ist(dt: Optional[datetime] = None, format_str: str = "%d %b %Y, %I:%M:%S %p IST") -> str:
    """
    Format a datetime object into a human-readable Indian Standard Time string.
    If dt is None, uses the current IST time.
    If dt is naive, assumes UTC and converts directly to IST.
    """
    if dt is None:
        dt = datetime.now(IST)
    elif dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc).astimezone(IST)
    else:
        dt = dt.astimezone(IST)
    return dt.strftime(format_str)

def to_ist(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert an arbitrary datetime to IST."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc).astimezone(IST)
    return dt.astimezone(IST)
