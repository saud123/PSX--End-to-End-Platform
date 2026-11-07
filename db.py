# db.py
"""
Unified database module — engine, schema, market-price loader, and
daily quant-snapshot refresher.

Responsibilities:
  • create / migrate tables (market_prices_eod, quant_metrics_snapshot, …)
  • fetch EOD prices from psxdata into market_prices_eod
  • compute quant metrics for the whole universe and STORE them in
    quant_metrics_snapshot (so the screener never has to fetch on request)
  • run that refresh on a schedule: every day at 17:45 PKT
  • delete the previous snapshot after the new one is written
"""

import os
import logging
import threading
import time
from datetime import datetime, timedelta, time as dtime
from typing import Optional, List, Dict, Any

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
_DB_URL = os.getenv("DATABASE_URL")
if _DB_URL:
    engine = create_engine(_DB_URL, pool_pre_ping=True, pool_recycle=1800,
                           pool_size=10, max_overflow=20)
else:
    engine = create_engine(
        URL.create(
            drivername="postgresql+psycopg2",
            username=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "psx_intelligence"),
        ),
        pool_pre_ping=True,
        pool_recycle=1800,
        pool_size=10,
        max_overflow=20,
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
def init_db() -> None:
    """Idempotent — safe to call on every boot."""
    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS pg_trgm;"))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS market_prices_eod (
                id SERIAL PRIMARY KEY,
                symbol VARCHAR(20) NOT NULL,
                trade_date DATE NOT NULL,
                open REAL, high REAL, low REAL, close REAL,
                volume BIGINT,
                UNIQUE(symbol, trade_date)
            );
            CREATE INDEX IF NOT EXISTS ix_prices_symbol_date
                ON market_prices_eod (symbol, trade_date DESC);
        """))

        # NEW: one row per (symbol, snapshot_date). Deleted after each refresh.
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS quant_metrics_snapshot (
                id SERIAL PRIMARY KEY,
                snapshot_date DATE NOT NULL,
                symbol VARCHAR(20) NOT NULL,
                metrics JSONB NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(snapshot_date, symbol)
            );
            CREATE INDEX IF NOT EXISTS ix_quant_snapshot_date
                ON quant_metrics_snapshot (snapshot_date DESC);
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id SERIAL PRIMARY KEY,
                headline TEXT,
                full_text TEXT,
                source VARCHAR(100),
                url TEXT,
                url_hash VARCHAR(64) UNIQUE,
                published_date TIMESTAMPTZ,
                sentiment VARCHAR(20) DEFAULT 'Neutral'
            );
            CREATE INDEX IF NOT EXISTS ix_news_published
                ON news_articles (published_date DESC);
            CREATE INDEX IF NOT EXISTS ix_news_headline_trgm
                ON news_articles USING gin (headline gin_trgm_ops);
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS macro_indicators_hub (
                id SERIAL PRIMARY KEY,
                series_id VARCHAR(100),
                indicator_name VARCHAR(255) NOT NULL,
                module VARCHAR(50),
                source VARCHAR(150),
                unit VARCHAR(50),
                frequency VARCHAR(30),
                indicator_value REAL,
                recorded_date DATE NOT NULL,
                dimensions TEXT,
                UNIQUE(indicator_name, recorded_date, dimensions)
            );
        """))

        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS finance_macro_dashboard (
                id SERIAL PRIMARY KEY,
                source_url TEXT,
                page_category TEXT,
                chart_title TEXT,
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (page_category, chart_title)
            );
        """))

    logger.info("✅ DB schema initialized")


# ---------------------------------------------------------------------------
# Market price loader
# ---------------------------------------------------------------------------
_OHLC_COLS = ("open", "high", "low", "close", "volume")


def _normalise_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()

    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    elif "trade_date" in out.columns:
        out["date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    else:
        return pd.DataFrame()

    for col in _OHLC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["date", "close"])
    out = out[out["close"] > 0]
    if "high" in out.columns:
        out = out[(out["high"].isna()) | (out["high"] >= out["close"])]
    if "low" in out.columns:
        out = out[(out["low"].isna()) | (out["low"] <= out["close"])]
    return out.sort_values("date").reset_index(drop=True)


def load_psx_market_prices(target_symbol: Optional[str] = None) -> None:
    """Fetch and upsert EOD prices via psxdata."""
    try:
        import psxdata
    except ImportError:
        logger.error("psxdata not installed — cannot load market prices")
        return

    current_date = datetime.today().date()
    symbols = [target_symbol.strip().upper()] if target_symbol else ["ENGRO", "OGDC", "LUCK", "KEL"]

    for symbol in symbols:
        try:
            with engine.connect() as conn:
                last = conn.execute(
                    text("SELECT MAX(trade_date) FROM market_prices_eod WHERE symbol = :s"),
                    {"s": symbol},
                ).scalar()

            if last and last >= current_date:
                logger.info("%s already up to date (%s)", symbol, last)
                continue

            start_date = (
                (last + timedelta(days=1)).strftime("%Y-%m-%d") if last
                else os.getenv("PSX_HISTORY_START", "2023-01-01")
            )

            df = psxdata.stocks(symbol, start=start_date, end=current_date.strftime("%Y-%m-%d"))
            df = _normalise_ohlc(df)
            if df.empty:
                logger.info("No new data for %s", symbol)
                continue

            rows = []
            for _, row in df.iterrows():
                date_str = row["date"].strftime("%Y-%m-%d")
                if date_str > current_date.strftime("%Y-%m-%d"):
                    continue
                close_val = float(row.get("close", 0) or 0)
                if close_val <= 0:
                    continue
                rows.append({
                    "symbol": symbol,
                    "trade_date": date_str,
                    "open":   float(row.get("open", 0) or 0),
                    "high":   float(row.get("high", close_val) or close_val),
                    "low":    float(row.get("low", close_val) or close_val),
                    "close":  close_val,
                    "volume": int(row.get("volume", 0) or 0),
                })

            if not rows:
                continue

            with engine.begin() as conn:
                conn.execute(
                    text("""
                        INSERT INTO market_prices_eod
                            (symbol, trade_date, open, high, low, close, volume)
                        VALUES
                            (:symbol, :trade_date, :open, :high, :low, :close, :volume)
                        ON CONFLICT (symbol, trade_date) DO UPDATE SET
                            open   = EXCLUDED.open,
                            high   = EXCLUDED.high,
                            low    = EXCLUDED.low,
                            close  = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """),
                    rows,
                )
            logger.info("Upserted %d rows for %s", len(rows), symbol)

        except Exception as exc:
            logger.exception("Loader failed for %s: %s", symbol, exc)


# ---------------------------------------------------------------------------
# NEW: snapshot read / write
# ---------------------------------------------------------------------------
def load_latest_quant_snapshot() -> Dict[str, Dict[str, Any]]:
    """
    Return {symbol: metrics_dict} from the most recent snapshot_date.
    Empty dict if the table has no rows yet.
    """
    out: Dict[str, Dict[str, Any]] = {}
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT symbol, metrics
                FROM quant_metrics_snapshot
                WHERE snapshot_date = (
                    SELECT MAX(snapshot_date) FROM quant_metrics_snapshot
                )
            """)).fetchall()
        for r in rows:
            out[r.symbol] = r.metrics
    except Exception as exc:
        logger.warning("load_latest_quant_snapshot failed: %s", exc)
    return out


def save_quant_snapshot(
    snapshot_date: datetime,
    metrics_map: Dict[str, Optional[Dict[str, Any]]],
    delete_older: bool = True,
) -> int:
    """
    Insert one row per symbol for snapshot_date.
    If delete_older=True, also delete every other snapshot_date in the
    SAME transaction, so a failed insert leaves the old snapshot intact.
    Returns number of rows inserted.
    """
    payload = [
        {"snapshot_date": snapshot_date.date(), "symbol": sym, "metrics": m}
        for sym, m in metrics_map.items()
        if m is not None
    ]
    if not payload:
        logger.warning("save_quant_snapshot: nothing to insert")
        return 0

    with engine.begin() as conn:  # atomic: all-or-nothing
        conn.execute(
            text("""
                INSERT INTO quant_metrics_snapshot (snapshot_date, symbol, metrics)
                VALUES (:snapshot_date, :symbol, CAST(:metrics AS JSONB))
                ON CONFLICT (snapshot_date, symbol)
                DO UPDATE SET metrics = EXCLUDED.metrics,
                              created_at = NOW()
            """),
            [
                {**p, "metrics": pd.io.json.dumps(p["metrics"])}
                for p in payload
            ],
        )
        if delete_older:
            conn.execute(
                text("""
                    DELETE FROM quant_metrics_snapshot
                    WHERE snapshot_date < :d
                """),
                {"d": snapshot_date.date()},
            )

    logger.info("💾 Snapshot saved: %d symbols for %s", len(payload), snapshot_date.date())
    return len(payload)


# ---------------------------------------------------------------------------
# NEW: refresh orchestration
# ---------------------------------------------------------------------------
def _refresh_all_prices(universe: List[str]) -> None:
    """Pull latest EOD rows from psxdata for each symbol (network-bound)."""
    try:
        import psxdata  # noqa: F401
    except ImportError:
        logger.warning("psxdata not installed — skipping price refresh")
        return

    for i, sym in enumerate(universe):
        try:
            load_psx_market_prices(target_symbol=sym)
        except Exception as exc:
            logger.exception("price refresh failed for %s: %s", sym, exc)
        if (i + 1) % 50 == 0:
            logger.info("  …price refresh %d/%d", i + 1, len(universe))


def _compute_metrics_from_db(universe: List[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Read EOD rows from Postgres in one query per 500 symbols and compute
    metrics with quant_screener.compute_metrics — no network.
    """
    from quant_screener import compute_metrics, _normalise_history, MIN_HISTORY_DAYS, FETCH_LOOKBACK_DAYS

    per_symbol: Dict[str, List[Dict[str, Any]]] = {}
    chunk = 500
    for i in range(0, len(universe), chunk):
        part = universe[i:i + chunk]
        with engine.connect() as conn:
            rows = conn.execute(text("""
                WITH ranked AS (
                    SELECT symbol, trade_date, open, high, low, close, volume,
                           ROW_NUMBER() OVER (
                               PARTITION BY symbol ORDER BY trade_date DESC
                           ) AS rn
                    FROM market_prices_eod
                    WHERE symbol = ANY(:symbols)
                )
                SELECT symbol, trade_date, open, high, low, close, volume
                FROM ranked WHERE rn <= :lim
                ORDER BY symbol, trade_date
            """), {"symbols": part, "lim": FETCH_LOOKBACK_DAYS}).fetchall()
        for r in rows:
            d = dict(r._mapping)
            per_symbol.setdefault(d["symbol"], []).append(d)

    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for sym in universe:
        rows = per_symbol.get(sym, [])
        if len(rows) < MIN_HISTORY_DAYS:
            out[sym] = None
            continue
        try:
            df = _normalise_history(pd.DataFrame(rows))
            out[sym] = compute_metrics(df) if len(df) >= MIN_HISTORY_DAYS else None
        except Exception as exc:
            logger.exception("metrics failed for %s: %s", sym, exc)
            out[sym] = None
    return out


def refresh_quant_snapshot(fetch_prices: bool = True) -> Dict[str, Any]:
    """
    Full daily job:
      1. (optional) pull new EOD rows from psxdata into market_prices_eod
      2. compute metrics for every universe symbol from Postgres
      3. write a fresh quant_metrics_snapshot and delete older snapshots
    """
    from quant_screener import get_universe  # local import avoids cycle at module load

    started = time.time()
    universe = get_universe()
    logger.info("🔄 Quant refresh starting for %d symbols", len(universe))

    if fetch_prices:
        _refresh_all_prices(universe)

    metrics_map = _compute_metrics_from_db(universe)
    inserted = save_quant_snapshot(datetime.utcnow(), metrics_map, delete_older=True)

    elapsed = time.time() - started
    logger.info("✅ Quant refresh done: %d metrics in %.1fs", inserted, elapsed)
    return {"inserted": inserted, "elapsed_sec": round(elapsed, 1)}


# ---------------------------------------------------------------------------
# NEW: daily scheduler at 17:45 (local / Pakistan time)
# ---------------------------------------------------------------------------
_REFRESH_AT = dtime(
    hour=int(os.getenv("PSX_REFRESH_HOUR", "17")),
    minute=int(os.getenv("PSX_REFRESH_MINUTE", "45")),
)

_DAILY_STARTED = False
_DAILY_LOCK = threading.Lock()


def _next_run_at(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.now()
    target = now.replace(
        hour=_REFRESH_AT.hour, minute=_REFRESH_AT.minute,
        second=0, microsecond=0,
    )
    if target <= now:
        target += timedelta(days=1)
    return target


def _daily_worker() -> None:
    """Sleep until the next 17:45, refresh, repeat."""
    while True:
        target = _next_run_at()
        sleep_sec = max(0.0, (target - datetime.now()).total_seconds())
        logger.info("⏰ Next quant refresh at %s (in %.1f h)", target, sleep_sec / 3600)
        time.sleep(sleep_sec)
        try:
            refresh_quant_snapshot(fetch_prices=True)
        except Exception:
            logger.exception("Scheduled quant refresh failed")


def start_daily_refresher() -> None:
    """Start the 17:45 daily refresh thread once. Safe to call repeatedly."""
    global _DAILY_STARTED
    with _DAILY_LOCK:
        if _DAILY_STARTED:
            return
        _DAILY_STARTED = True
    threading.Thread(target=_daily_worker, name="db-quant-refresher", daemon=True).start()
    logger.info("📅 Daily quant refresher started (target %02d:%02d)",
                _REFRESH_AT.hour, _REFRESH_AT.minute)


# ---------------------------------------------------------------------------
# Boot
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    init_db()
    start_daily_refresher()
    # keep the process alive when run directly
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        pass