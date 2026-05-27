from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

import requests
import yfinance as yf
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from libb.execution.get_market_data import download_data_on_given_date
from ..prompt_orchestration.get_prompt_data.fetching import fmp_endpoint
from ..prompt_orchestration.get_prompt_data.utilities import safe_float

TODAY = _dt.date.today()

from dotenv import load_dotenv
load_dotenv()

MINIMUM_MARKET_CAP = 200_000_000
IPO_LOCKOUT_YEARS = 3
MINIMUM_AVG_VOLUME = 1_000_000

POLYGON_API_KEY = (
    os.getenv("POLYGON_API_KEY")
    or os.getenv("MASSIVE_API_KEY")
)
FMP_API_KEY = os.getenv("FMP_API_KEY") or os.getenv("MASSIVE_API_KEY")

POLYGON_BASE_URL = "https://api.polygon.io"
FMP_BASE_URL = "https://financialmodelingprep.com"

REQUEST_TIMEOUT = 15

SPAC_PATTERNS = (
    r"\bspac\b",
    r"\bblank check\b",
    r"\bacquisition corp\b",
    r"\bacquisition corporation\b",
    r"\bcapital acquisition\b",
    r"\bmerger corp\b",
)

# =========================================================
# CACHE
# =========================================================

_TICKER_CACHE: dict[str, dict] = {}
_CACHE_LOCK = Lock()
session = requests.Session()

retry = Retry(
    total=3,
    backoff_factor=0.4,
    status_forcelist=(429, 500, 502, 503, 504),
    allowed_methods=frozenset(["GET"]),
)

session.mount("https://", HTTPAdapter(max_retries=retry))
session.mount("http://", HTTPAdapter(max_retries=retry))

CACHE_PATH = Path(
    os.getenv(
        "ORDER_FILTER_CACHE_PATH",
        str(Path(__file__).with_suffix(".cache.sqlite3")),
    )
)

DB_LOCK = Lock()


def _init_db() -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(CACHE_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS snapshot_cache (
                cache_key TEXT PRIMARY KEY,
                payload TEXT NOT NULL,
                fetched_at INTEGER NOT NULL
            )
            """
        )
        conn.commit()


_init_db()


# =========================================================
# HELPERS
# =========================================================


def _parse_date(value: Any) -> _dt.date | None:
    if value is None or value == "":
        return None

    if isinstance(value, _dt.date) and not isinstance(value, _dt.datetime):
        return value

    if isinstance(value, _dt.datetime):
        return value.date()

    if isinstance(value, (int, float)):
        try:
            # Assume UNIX seconds.
            return _dt.date.fromtimestamp(float(value))
        except Exception:
            return None

    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None

        # Handle ISO-ish strings and common datetime formats.
        candidates = (
            raw,
            raw.split("T", 1)[0],
            raw.split(" ", 1)[0],
        )
        for candidate in candidates:
            try:
                return _dt.date.fromisoformat(candidate)
            except Exception:
                continue

        try:
            return _dt.datetime.fromisoformat(raw.replace("Z", "+00:00")).date()
        except Exception:
            return None

    return None


def _days_since(value: Any, today: _dt.date) -> int | None:
    d = _parse_date(value)
    if d is None:
        return None
    return (today - d).days


# =========================================================
# VALIDATION
# =========================================================

def _calculate_market_cap(
    order: dict,
) -> float:
    
    ticker = order.get("ticker")

    if ticker is None:
        return 0.0
    
    share_count_data = fmp_endpoint("shares-float", ticker)
    share_count = share_count_data["outstandingShares"]

    order_type = (order.get("order_type", "MARKET") or "MARKET").upper()
    limit_price = safe_float(order.get("limit_price")) or 0.0

    if order_type == "LIMIT":
        return share_count * limit_price

    # assume market
    else:
        try:
            ticker_data = download_data_on_given_date(ticker, TODAY)
            price = safe_float(ticker_data.get("Open"))
        except Exception:
            price = 0.0

        return share_count * price
    
def _get_ipo_date(
    order: dict,
) -> str:
    ticker = order.get("ticker", None)

    if ticker is None:
        return "UNKNOWN"

    ticker_data = fmp_endpoint("profile", ticker)
    ipo_date = ticker_data.get("ipoDate", None)
    if ipo_date is None:
        return "UNKNOWN"
    return ipo_date

def _get_rejection_reasons(
    order: dict,
) -> list[str]:
    reasons: list[str] = []

    # =====================================================
    # IPO CHECK
    # =====================================================

    ipo_date = _get_ipo_date(order)
    if ipo_date:
        age_days = _days_since(ipo_date, TODAY)
        if age_days is not None:
            age_years = age_days / 365.25
            if age_years > IPO_LOCKOUT_YEARS:
                reasons.append(
                    f"IPO too old ({age_years:.1f} yrs > {IPO_LOCKOUT_YEARS})"
                )
    else:
        reasons.append("IPO date unknown — cannot verify age")

    # =====================================================
    # MARKET CAP CHECK
    # =====================================================

    market_cap = _calculate_market_cap(order)
    if market_cap < MINIMUM_MARKET_CAP:
        reasons.append(
            f"market cap too low (${market_cap:,.0f} < ${MINIMUM_MARKET_CAP:,.0f})"
        )
    elif market_cap == 0.0:
        reasons.append("Market cap unknown")

    return reasons


# =========================================================
# MAIN FILTER
# =========================================================

def filter_orders(
    orders: dict,
    max_workers: int = 5,
) -> tuple[dict| None, list[dict] | None]:
    order_list = orders.get("orders", [])

    filtered_orders = []
    rejected_orders = []

    # =====================================================
    # PRELOAD UNIQUE TICKERS DETERMINISTICALLY
    # =====================================================

    # =====================================================
    # FILTER LOOP
    # =====================================================

    def _process_order(order: dict) -> tuple[dict, str]:

        action = order.get("action")
        if action != "b":
            return order, "pass"

        reasons = _get_rejection_reasons(order)
        if reasons:
            return {**order, "rejection_reasons": reasons}, "reject"
        return order, "pass"

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = []
        for order in order_list:
            future = ex.submit(_process_order, order)
            futures.append(future)
    
        for fut in as_completed(futures):
            order, status = fut.result()
            if status == "reject":
                rejected_orders.append(order)
            else:
                filtered_orders.append(order)

        filtered_orders = {"orders": filtered_orders}

    return filtered_orders, rejected_orders


if __name__ == "__main__":
    sample = {
        "orders": [
            {"ticker": "SNOW", "action": "b", "order_type": "MARKET"},
            {"ticker": "MDB", "action": "b", "order_type": "LIMIT", "limit_price": 300},
        ]
    }

    filtered, rejected = filter_orders(sample)
    print(filtered)
    print(rejected)
