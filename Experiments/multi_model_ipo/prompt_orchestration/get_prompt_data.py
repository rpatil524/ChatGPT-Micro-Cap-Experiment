from __future__ import annotations

import json
import os
import random
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests
import yfinance as yf
from dateutil.relativedelta import relativedelta
from pandas import Series
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

REQUEST_TIMEOUT = 20

CACHE_PATH = Path(".cache/ipo_cache.sqlite3")

POLYGON_TTL_SECONDS = 24 * 3600
FMP_TTL_SECONDS = 6 * 3600

MIN_MARKET_CAP = 200_000_000
MIN_DOLLAR_VOLUME = 5_000_000

DEFAULT_LOOKBACK_YEARS = 3
DEFAULT_MAX_RESULTS = 25
DEFAULT_MAX_WORKERS = 3

# =========================================================
# HTTP SESSION
# =========================================================

session = requests.Session()

retry = Retry(
    total=3,
    backoff_factor=0.5,
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

# =========================================================
# HELPERS
# =========================================================

def parse_date(x):
    try:
        return pd.to_datetime(x).date()
    except:
        return None

def looks_like_spac(name: str, description: str):
    blob = f"{name} {description}".lower()

    keywords = [
        "blank check",
        "spac",
        "acquisition corp",
        "special purpose acquisition company",
    ]

    return any(k in blob for k in keywords)


def looks_shellish(name: str, description: str):
    blob = f"{name} {description}".lower()

    keywords = [
        "shell",
        "holding company",
        "exploration stage",
    ]

    return any(k in blob for k in keywords)


def looks_biotech(name: str, description: str):
    blob = f"{name} {description}".lower()

    keywords = [
        "biotech",
        "pharma",
        "therapeutics",
        "clinical-stage",
        "drug",
    ]

    return any(k in blob for k in keywords)


def cache_key(provider, url, params):
    return f"{provider}:{url}:{json.dumps(params, sort_keys=True)}"


# =========================================================
# REQUEST
# =========================================================


def request_json(provider, url, params, ttl, api_key):
    key = cache_key(provider, url, params)

    cached = CACHE.get(key, ttl)

    if cached is not None:
        return cached

    if not api_key:
        return None

    params = dict(params)

    if provider == "polygon":
        params["apiKey"] = api_key

    elif provider == "fmp":
        params["apikey"] = api_key

    try:
        r = session.get(url, params=params, timeout=REQUEST_TIMEOUT)

        r.raise_for_status()

        data = r.json()

        CACHE.set(key, data)

        return data

    except Exception:
        return None


# =========================================================
# POLYGON
# =========================================================


def get_ipos(start: str, end: str):
    data = request_json(
        "polygon",
        f"{POLYGON_BASE_URL}/vX/reference/ipos",
        {
            "listing_date.gte": start,
            "listing_date.lte": end,
            "ipo_status": "history",
            "limit": 1000,
        },
        POLYGON_TTL_SECONDS,
        POLYGON_API_KEY,
    )

    if not isinstance(data, dict):
        return []

    return data.get("results", [])


def get_polygon_details(ticker: str):
    ticker = normalize_ticker(ticker)

    data = request_json(
        "polygon",
        f"{POLYGON_BASE_URL}/v3/reference/tickers/{ticker}",
        {},
        POLYGON_TTL_SECONDS,
        POLYGON_API_KEY,
    )

    if not isinstance(data, dict):
        return {}

    return data.get("results", {})


# =========================================================
# FMP
# =========================================================


def fmp_endpoint(path: str, ticker: str):
    data = request_json(
        "fmp",
        f"{FMP_BASE_URL}/{path}/{ticker}",
        {},
        FMP_TTL_SECONDS,
        FMP_API_KEY,
    )

    if isinstance(data, list):
        return data[0] if data else {}

    if isinstance(data, dict):
        return data

    return {}


def get_fmp_data(ticker: str):
    profile = fmp_endpoint("profile", ticker)
    quote = fmp_endpoint("quote", ticker)
    income = fmp_endpoint("income-statement", ticker)
    balance = fmp_endpoint("balance-sheet-statement", ticker)
    cashflow = fmp_endpoint("cash-flow-statement", ticker)

    return {
        "profile": profile,
        "quote": quote,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
    }


# =========================================================
# MARKET DATA
# =========================================================


def get_market_data(ticker: str):
    try:
        yf_ticker = yf.Ticker(ticker)

        hist = yf_ticker.history(period="6mo")

        if hist.empty:
            return {}

        px = float(hist["Close"].iloc[-1])

        avg_volume = float(hist["Volume"].tail(20).mean())

        dollar_volume = px * avg_volume

        atr = float((hist["High"] - hist["Low"]).tail(14).mean())

        mom_1m = (
            (px / float(hist["Close"].iloc[-21])) - 1
            if len(hist) >= 21
            else 0
        )

        mom_3m = (
            (px / float(hist["Close"].iloc[-63])) - 1
            if len(hist) >= 63
            else 0
        )

        return {
            "price": round(px, 2),
            "avg_volume": int(avg_volume),
            "dollar_volume": int(dollar_volume),
            "atr": round(atr, 2),
            "mom_1m": round(mom_1m * 100, 2),
            "mom_3m": round(mom_3m * 100, 2),
        }

    except:
        return {}


# =========================================================
# ENRICHMENT
# =========================================================


def enrich_company(ticker: str, ipo_row: dict):
    details = get_polygon_details(ticker)

    if not details:
        return None

    listing_date = (
        details.get("list_date")
        or details.get("ipo_date")
        or ipo_row.get("listing_date")
    )

    parsed_listing = parse_date(listing_date)

    # CRITICAL FIX:
    # EXCLUDE FUTURE IPOS
    if parsed_listing is None or parsed_listing > date.today():
        return None

    market_cap = safe_float(details.get("market_cap"))

    if market_cap is None:
        return None

    if market_cap < MIN_MARKET_CAP:
        return None

    name = details.get("name", ticker)

    description = details.get("description", "")

    # FILTER JUNK
    if looks_like_spac(name, description):
        return None

    if looks_shellish(name, description):
        return None

    fmp = get_fmp_data(ticker)

    market = get_market_data(ticker)

    # LIQUIDITY FILTER
    if market.get("dollar_volume", 0) < MIN_DOLLAR_VOLUME:
        return None

    flags = []

    if looks_biotech(name, description):
        flags.append("BIOTECH")

    return {
        "ticker": ticker,
        "name": name,
        "description": description,
        "listing_date": str(parsed_listing),
        "market_cap": market_cap,
        "sector": (
            details.get("sic_description")
            or fmp["profile"].get("sector")
            or "Unknown"
        ),
        "flags": flags,
        "price": market.get("price"),
        "avg_volume": market.get("avg_volume"),
        "dollar_volume": market.get("dollar_volume"),
        "atr": market.get("atr"),
        "mom_1m": market.get("mom_1m"),
        "mom_3m": market.get("mom_3m"),
        "revenue": safe_float(fmp["income"].get("revenue")),
        "net_income": safe_float(fmp["income"].get("netIncome")),
        "cash": safe_float(
            fmp["balance"].get("cashAndCashEquivalents")
        ),
        "debt": safe_float(fmp["balance"].get("totalDebt")),
        "ocf": safe_float(
            fmp["cashflow"].get("operatingCashFlow")
        ),
    }


# =========================================================
# MAIN UNIVERSE
# =========================================================


def get_ipo_universe(
    lookback_years=3,
    max_results=25,
):
    end = date.today()

    start = end - relativedelta(years=lookback_years)

    ipos = get_ipos(str(start), str(end))

    if not ipos:
        return []

    dedup = {}

    for ipo in ipos:
        ticker = normalize_ticker(ipo.get("ticker"))

        if not ticker:
            continue

        dedup[ticker] = ipo

    ipo_rows = sorted(
        dedup.items(),
        key=lambda x: x[0],
    )

    results = []

    with ThreadPoolExecutor(max_workers=DEFAULT_MAX_WORKERS) as ex:
        futures = {
            ex.submit(enrich_company, ticker, row): ticker
            for ticker, row in ipo_rows
        }

        for fut in as_completed(futures):
            result = fut.result()

            if result:
                results.append(result)

    # BETTER RANKING
    def score(x):
        mcap = x.get("market_cap", 0)

        liquidity = x.get("dollar_volume", 0)

        momentum = x.get("mom_3m", 0)

        return (
            (mcap / 1e9)
            + (liquidity / 1e7)
            + momentum
        )

    results.sort(key=score, reverse=True)

    return results[:max_results]


def get_polygon_ticker_details(ticker: str) -> dict:
    """
    Alias used by build_eligibility_series.
    Wraps get_polygon_details with a fallback to empty dict.
    """
    ticker = normalize_ticker(ticker)
    if not ticker:
        return {}
    return get_polygon_details(ticker) or {}


def pull_polygon_listing_date(details: dict, ipo_row: dict | None) -> str | None:
    """
    Extract listing/IPO date from Polygon details dict,
    falling back to ipo_row if present.

    Returns a raw string (or None) — caller is responsible for parsing.
    """
    if not details:
        return (ipo_row or {}).get("listing_date")

    return (
        details.get("list_date")
        or details.get("ipo_date")
        or (ipo_row or {}).get("listing_date")
    )


def pull_polygon_market_cap(details: dict) -> float | None:
    """
    Extract market cap from Polygon details dict.
    Tries both field names Polygon has been known to use.
    Returns float or None.
    """
    if not details:
        return None

    return safe_float(
        details.get("market_cap")
        or details.get("mktCap")
    )


# =========================================================
# FORMATTER
# =========================================================


def fmt_billions(x):
    if x is None:
        return "UNKNOWN"

    return f"{x / 1e9:.2f}B"


def fmt_millions(x):
    if x is None:
        return "UNKNOWN"

    return f"{x / 1e6:.1f}M"


def format_universe_for_prompt(companies):
    lines = ["IPO_UNIVERSE_START"]

    for c in companies:
        lines.append(
            f"TICKER={c['ticker']} | "
            f"NAME={c['name']} | "
            f"IPO={c['listing_date']} | "
            f"MCAP={fmt_billions(c['market_cap'])} | "
            f"PX={c['price']} | "
            f"ATR={c['atr']} | "
            f"VOL={fmt_millions(c['avg_volume'])} | "
            f"DOLLAR_VOL={fmt_millions(c['dollar_volume'])} | "
            f"MOM_1M={c['mom_1m']}% | "
            f"MOM_3M={c['mom_3m']}% | "
            f"SECTOR={c['sector']} | "
            f"FLAGS={','.join(c['flags']) if c['flags'] else 'NONE'} | "
            f"FIN=revenue:{fmt_billions(c['revenue'])}, "
            f"net_income:{fmt_billions(c['net_income'])}, "
            f"cash:{fmt_billions(c['cash'])}, "
            f"debt:{fmt_billions(c['debt'])}, "
            f"operating_cash_flow:{fmt_billions(c['ocf'])} | "
            f"DESC={truncate(c['description'])}"
        )

    lines.append("IPO_UNIVERSE_END")

    return "\n".join(lines)

# =========================================================
# CORE SAFE UTILITIES
# =========================================================

def safe_float(x):
    try:
        if x is None or x == "":
            return None
        return float(x)
    except:
        return None


def safe_int(x):
    try:
        if x is None or x == "":
            return None
        return int(float(x))
    except:
        return None


def first_nonempty(*vals):
    for v in vals:
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        return v
    return None


def parse_date_safe(x):
    try:
        return pd.to_datetime(x).date()
    except:
        return None


# =========================================================
# IPO + MARKET VALIDATION LOGIC
# =========================================================

def is_valid_ipo_date(ipo_date, today):
    if ipo_date is None:
        return False
    if ipo_date > today:
        return False
    return True


def passes_market_cap(mcap, min_mcap):
    return mcap is not None and mcap >= min_mcap


def passes_liquidity(price, avg_vol, min_dollar_vol):
    if price is None or avg_vol is None:
        return False
    return (price * avg_vol) >= min_dollar_vol


# =========================================================
# TICKER HANDLING
# =========================================================

def normalize_ticker(t):
    if not t:
        return ""
    return str(t).upper().strip()


def dedupe_tickers(items):
    seen = set()
    out = []
    for x in items:
        t = normalize_ticker(x)
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


# =========================================================
# IPO DATA HELPERS
# =========================================================

def get_listing_date(details, ipo_row):
    return first_nonempty(
        (details or {}).get("list_date"),
        (details or {}).get("ipo_date"),
        (ipo_row or {}).get("listing_date")
    )


def extract_market_cap(details):
    return safe_float(
        first_nonempty(
            (details or {}).get("market_cap"),
            (details or {}).get("mktCap")
        )
    )


# =========================================================
# SCORING + RANKING
# =========================================================

def score_company(c):
    mcap = c.get("market_cap") or 0
    liq = c.get("dollar_volume") or 0
    mom = c.get("momentum") or 0

    return (mcap / 1e9) + (liq / 1e7) + mom


# =========================================================
# FORMATTING
# =========================================================

def format_universe_line(c):
    return (
        f"TICKER={c.get('ticker')} | "
        f"MCAP={c.get('market_cap')} | "
        f"VOL={c.get('avg_volume')} | "
        f"MOM={c.get('momentum')} | "
        f"FLAGS={','.join(c.get('flags', [])) or 'NONE'}"
    )


# =========================================================
# ELIGIBILITY REPORT (FIXED + SIMPLE)
# =========================================================

from datetime import date
from typing import Iterable
import pandas as pd


def build_eligibility_series(
    tickers,
    get_polygon_details=None,
    min_mcap=200_000_000,
    years_back=3,
):
    """
    Flexible overload-style wrapper.

    Supports:
    - pandas Series
    - list[str]
    - tuple[str]
    - set[str]
    - single ticker string

    Usage:
        build_eligibility_series(df["ticker"])

        build_eligibility_series(
            df["ticker"],
            get_polygon_details=_get_polygon_ticker_details,
            min_mcap=MIN_MARKET_CAP
        )
    """

    # ---------------------------------------------------------
    # normalize input
    # ---------------------------------------------------------

    if tickers is None:
        return ""

    if isinstance(tickers, str):
        tickers = [tickers]

    elif isinstance(tickers, pd.Series):
        tickers = tickers.dropna().astype(str).tolist()

    elif not isinstance(tickers, Iterable):
        tickers = [str(tickers)]

    # ---------------------------------------------------------
    # defaults
    # ---------------------------------------------------------

    if get_polygon_details is None:
        get_polygon_details = get_polygon_ticker_details

    today = date.today()
    cutoff = today.replace(year=today.year - years_back)

    lines = []

    # ---------------------------------------------------------
    # main logic
    # ---------------------------------------------------------

    for t in tickers:
        t = normalize_ticker(t)

        if not t:
            continue

        data = get_polygon_details(t)

        if not data:
            lines.append(f"{t} | NO_DATA | BUY_BLOCKED | NOT_ELIGIBLE")
            continue

        listing = parse_date(
            pull_polygon_listing_date(data, None)
        )

        if not listing:
            lines.append(f"{t} | NO_DATE | BUY_BLOCKED | NOT_ELIGIBLE")
            continue

        ipo_ok = listing >= cutoff
        if ipo_ok:
            eligibility = pd.Timestamp(cutoff) - pd.Timestamp(listing)
            eligibility = str(eligibility)
        else:
            eligibility = "NOT_ELIGIBLE"

        mcap = pull_polygon_market_cap(data)
        mcap_ok = (
            isinstance(mcap, (int, float))
            and mcap >= min_mcap
        )

        status = (
            "BUY_ALLOWED"
            if (ipo_ok and mcap_ok)
            else "BUY_BLOCKED"
        )

        lines.append(
            f"{t} | "
            f"IPO_OK={ipo_ok} | "
            f"MCAP={mcap} | "
            f"DAYS_ELIGIBILITY_LEFT={eligibility}"
            f"{status}"
        )

    return "\n".join(lines)

def build_eligibility_series_from_universe(universe: list[dict]) -> str:
    """
    Takes the already-enriched list of companies from get_ipo_universe()
    and builds an eligibility report string without making any additional
    API calls — all required data is already on each company dict.
    """
    if not universe:
        return ""

    today = date.today()
    cutoff = today - relativedelta(years=DEFAULT_LOOKBACK_YEARS)

    lines = []

    for c in universe:
        ticker = normalize_ticker(c.get("ticker", ""))

        if not ticker:
            continue

        listing = parse_date(c.get("listing_date"))
        ipo_ok = listing is not None and listing >= cutoff

        mcap = safe_float(c.get("market_cap"))
        mcap_ok = mcap is not None and mcap >= MIN_MARKET_CAP

        liq_ok = (c.get("dollar_volume") or 0) >= MIN_DOLLAR_VOLUME

        if ipo_ok and mcap_ok and liq_ok:
            status = "BUY_ALLOWED"
        else:
            reasons = []
            if not ipo_ok:
                reasons.append("IPO_OUT_OF_WINDOW")
            if not mcap_ok:
                reasons.append("MCAP_TOO_SMALL")
            if not liq_ok:
                reasons.append("ILLIQUID")
            status = "BUY_BLOCKED:" + "+".join(reasons)

        lines.append(
            f"{ticker} | "
            f"IPO={c.get('listing_date')} | "
            f"IPO_OK={ipo_ok} | "
            f"MCAP={fmt_billions(mcap)} | "
            f"MCAP_OK={mcap_ok} | "
            f"DOLLAR_VOL={fmt_millions(c.get('dollar_volume'))} | "
            f"LIQ_OK={liq_ok} | "
            f"{status}"
        )

    return "\n".join(lines)

import yfinance as yf
from datetime import date


def truncate(text, limit=200):
    if not text:
        return ""
    text = str(text).strip()
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "..."


def get_macro_news(n=5):
    """
    PURE DATA LAYER ONLY.

    No interpretation.
    No sentiment.
    No labeling.

    Goal:
    Feed LLM raw macro + market state so it can infer structure itself.
    """

    lines = []

    # -------------------------------------------------
    # SPY RAW MARKET DATA
    # -------------------------------------------------

    try:
        spy = yf.Ticker("SPY")
        hist = spy.history(period="5d")

        if len(hist) > 0:
            latest = hist.iloc[-1]

            lines.append(
                f"SPY_CLOSE={float(latest['Close'])} "
                f"SPY_VOLUME={float(latest['Volume'])}"
            )
    except:
        lines.append("SPY=ERROR")

    # -------------------------------------------------
    # VIX RAW DATA
    # -------------------------------------------------

    try:
        vix = yf.Ticker("^VIX")
        vh = vix.history(period="5d")

        if len(vh) > 0:
            latest = vh.iloc[-1]
            lines.append(f"VIX_CLOSE={float(latest['Close'])}")
    except:
        lines.append("VIX=ERROR")

    # -------------------------------------------------
    # TREASURY YIELD (10Y proxy via ^TNX)
    # -------------------------------------------------

    try:
        tn = yf.Ticker("^TNX")
        th = tn.history(period="5d")

        if len(th) > 0:
            latest = th.iloc[-1]
            lines.append(f"TNX_10Y={float(latest['Close'])}")
    except:
        lines.append("TNX=ERROR")

    # -------------------------------------------------
    # MACRO HEADLINES (RAW ONLY)
    # -------------------------------------------------

    try:
        news = yf.Ticker("^GSPC").news or []
    except:
        news = []

    lines.append("HEADLINES_START")

    for item in news[:n]:
        content = item.get("content", {}) if isinstance(item, dict) else {}

        title = content.get("title") or ""
        summary = content.get("summary") or item.get("summary") or ""

        lines.append(
            f"- {title} | {truncate(summary, 180)}"
        )

    lines.append("HEADLINES_END")

    # -------------------------------------------------
    # RETURN RAW BLOCK
    # -------------------------------------------------

    return "MACRO_RAW_START\n" + "\n".join(lines) + "\nMACRO_RAW_END"

# =========================================================
# TEST
# =========================================================

if __name__ == "__main__":
    universe = get_ipo_universe(
        lookback_years=3,
        max_results=15,
    )

    print(format_universe_for_prompt(universe))