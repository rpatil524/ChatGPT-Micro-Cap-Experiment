from __future__ import annotations

import json
import os
import sqlite3
import threading
from threading import Lock
import time
from pathlib import Path
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .utilities import *
from .fetching import *

from dotenv import load_dotenv
load_dotenv()

# =========================================================
# CONFIG
# =========================================================

POLYGON_API_KEY = (
    os.getenv("MASSIVE_API_KEY")
    or os.getenv("POLYGON_API_KEY")
    or os.getenv("POLYGON_KEY")
)

FMP_API_KEY = (
    os.getenv("FMP_API_KEY")
    or os.getenv("FINANCIAL_MODELING_PREP_API_KEY")
    or os.getenv("FINANCIAL_MODELING_PREP_KEY")
)

POLYGON_BASE_URL = "https://api.polygon.io"

# FIXED:
FMP_BASE_URL = "https://financialmodelingprep.com"

REQUEST_TIMEOUT = 20

CACHE_PATH = Path(".cache/ipo_cache.sqlite3")

POLYGON_TTL_SECONDS = 24 * 3600
FMP_TTL_SECONDS = 6 * 3600

MIN_MARKET_CAP = 200_000_000
MIN_DOLLAR_VOLUME = 5_000_000

DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_MAX_RESULTS = 25
DEFAULT_MAX_WORKERS = 2

POLYGON_MIN_INTERVAL = 12.5  # 5 req/min
FMP_MIN_INTERVAL = 0       # adjust based on free tier

_last_polygon_call = 0.0
_last_fmp_call = 0.0

POLYGON_LOCK = Lock()
FMP_LOCK = Lock()

# =========================================================
# HTTP SESSION
# =========================================================

session = requests.Session()

retry = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)

adapter = HTTPAdapter(max_retries=retry)

session.mount("https://", adapter)
session.mount("http://", adapter)

# =========================================================
# CACHE
# =========================================================


class SQLiteCache:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()

        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    created REAL,
                    payload TEXT
                )
                """
            )

    def get(self, key: str, ttl: int):
        with self.lock:
            row = self.conn.execute(
                "SELECT created, payload FROM cache WHERE key=?",
                (key,),
            ).fetchone()

        if not row:
            return None

        created, payload = row

        if (time.time() - created) > ttl:
            return None

        try:
            return json.loads(payload)
        except:
            return None

    def set(self, key: str, value: Any):
        payload = json.dumps(value, default=str)

        with self.lock:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO cache
                VALUES (?, ?, ?)
                """,
                (key, time.time(), payload),
            )
            self.conn.commit()


CACHE = SQLiteCache(CACHE_PATH)