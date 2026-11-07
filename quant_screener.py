# quant_screener.py
"""
Pure-quantitative PSX "what should I buy TODAY" screener.
No LLM. No model training. Pure math over historical prices.

Data flow (NEW):
  • db.py owns the nightly refresh at 17:45 → writes quant_metrics_snapshot
  • this module READS that snapshot (one SELECT), applies filters + scoring
    in pandas, and returns the result — no network on the request path.
  • If the snapshot table is empty, falls back to computing from
    market_prices_eod (still no network). psxdata is only used by db.py.
"""

import os
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sqlalchemy import text
from dotenv import load_dotenv

# ── CHANGED: also import the snapshot readers + refresher entrypoints
from db import (
    engine,
    load_psx_market_prices,
    load_latest_quant_snapshot,
    refresh_quant_snapshot,
    start_daily_refresher as _db_start_daily_refresher,
    init_db,
)

load_dotenv()

try:
    import psxdata
    HAS_PSXDATA = True
except ImportError:
    psxdata = None
    HAS_PSXDATA = False
    print("ℹ️ psxdata not installed — DB-only mode. Install with: pip install psxdata")


# ============================================================
# Config  (unchanged)
# ============================================================
FETCH_LOOKBACK_DAYS = 400
MIN_HISTORY_DAYS = 220
MAX_DATA_AGE_CALENDAR_DAYS = int(os.getenv("PSX_MAX_DATA_AGE_DAYS", "7"))
RISK_FREE_RATE_ANNUAL = float(os.getenv("PSX_RISK_FREE_RATE", "0.10"))

FILTER_MIN_PRICE              = 10.0
FILTER_MIN_AVG_DAILY_PKR      = 5_000_000.0
FILTER_MAX_RSI                = 75.0
FILTER_MIN_RSI                = 30.0
FILTER_MAX_PCT_ABOVE_SMA20    = 12.0
FILTER_MAX_PCT_BELOW_52W_HIGH = 40.0
FILTER_MAX_VOL_ANN_PCT        = 60.0
FILTER_MIN_RR_RATIO           = 1.5

DEFAULT_TOP_N = 10

PROFILES: Dict[str, Dict[str, float]] = {
    "balanced": {
        "entry_positioning":   0.22, "risk_reward": 0.25,
        "momentum_health":     0.20, "trend_structure": 0.15,
        "liquidity":           0.10, "volume_confirmation": 0.08,
    },
    "aggressive": {
        "entry_positioning":   0.15, "risk_reward": 0.20,
        "momentum_health":     0.30, "trend_structure": 0.20,
        "liquidity":           0.08, "volume_confirmation": 0.07,
    },
    "conservative": {
        "entry_positioning":   0.28, "risk_reward": 0.32,
        "momentum_health":     0.10, "trend_structure": 0.15,
        "liquidity":           0.10, "volume_confirmation": 0.05,
    },
}

# ── CHANGED: metrics cache is now optional — snapshot is the primary source.
#    Kept for the fallback path and for backwards-compat imports.
_METRICS_CACHE: Dict[str, Any] = {"ts": 0.0, "data": {}, "universe_key": None}
_RESULT_CACHE: Dict[str, Any] = {}
_RESULT_CACHE_MAX_ENTRIES = 32
_RESULT_CACHE_TTL_SECONDS = 1800
_CACHE_TTL_SECONDS = _RESULT_CACHE_TTL_SECONDS

_UNIVERSE_CACHE: Dict[str, Any] = {"ts": 0.0, "data": []}
_UNIVERSE_CACHE_TTL = 3600

_PSXDATA_LOCK = threading.Lock()


# ============================================================
# Noise suppression + universe hygiene  (unchanged)
# ============================================================
def _quiet_psxdata(fn, *args, **kwargs):
    with _PSXDATA_LOCK:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        out_buf, err_buf = io.StringIO(), io.StringIO()
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            return fn(*args, **kwargs)


_NON_EQUITY_SUBSTRINGS = ("TB", "TFC", "PIB", "FRR", "VRR", "PFL", "SFO", "STSC", "ETF")
_NON_EQUITY_SUFFIXES = ("SC", "TFC", "R")


def _is_equity_symbol(sym: str) -> bool:
    if not sym:
        return False
    s = sym.upper().strip()
    if len(s) < 2 or len(s) > 6:
        return False
    if any(p in s for p in _NON_EQUITY_SUBSTRINGS):
        return False
    if any(s.endswith(p) for p in _NON_EQUITY_SUFFIXES):
        return False
    return True


# ============================================================
# Result-cache helper  (unchanged)
# ============================================================
def _result_cache_put(result_key: str, result: Dict[str, Any]) -> None:
    _RESULT_CACHE[result_key] = {"ts": time.time(), "result": result}
    if len(_RESULT_CACHE) > _RESULT_CACHE_MAX_ENTRIES:
        oldest = min(_RESULT_CACHE, key=lambda k: _RESULT_CACHE[k]["ts"])
        if oldest != result_key:
            _RESULT_CACHE.pop(oldest, None)


# ============================================================
# Universe  (unchanged)
# ============================================================
def get_universe() -> List[str]:
    now = time.time()
    if _UNIVERSE_CACHE["data"] and now - _UNIVERSE_CACHE["ts"] < _UNIVERSE_CACHE_TTL:
        return list(_UNIVERSE_CACHE["data"])

    symbols = set()
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT symbol FROM market_prices_eod")).fetchall()
        symbols.update(str(r[0]).upper() for r in rows if r[0])
    except Exception as e:
        print(f"⚠️ DB symbol query failed: {e}")

    if HAS_PSXDATA and psxdata is not None:
        try:
            raw = _quiet_psxdata(psxdata.symbols)
            parsed: List[str] = []
            if raw is None:
                pass
            elif isinstance(raw, list):
                parsed = [str(x) for x in raw]
            elif hasattr(raw, "empty"):
                if not raw.empty:
                    if hasattr(raw, "columns"):
                        for c in ("symbol","Symbol","SYMBOL","ticker","Ticker","TICKER","code","Code","CODE"):
                            if c in raw.columns:
                                parsed = [str(x) for x in raw[c].tolist()]
                                break
                        else:
                            parsed = [str(x) for x in raw.iloc[:, 0].tolist()]
                    else:
                        parsed = [str(x) for x in raw.tolist()]
            elif isinstance(raw, dict):
                parsed = [str(x) for x in raw.values()]
            else:
                try:
                    parsed = [str(x) for x in list(raw)]
                except Exception:
                    parsed = []
            symbols.update(s.upper().strip() for s in parsed if s and s.strip())
        except Exception as e:
            print(f"⚠️ psxdata.symbols() failed: {e}")

    equities = sorted({s for s in symbols if _is_equity_symbol(s)})
    _UNIVERSE_CACHE["ts"] = now
    _UNIVERSE_CACHE["data"] = equities
    return equities


# ============================================================
# History normalisation + fetch  (unchanged — used by the DB fallback)
# ============================================================
def _normalise_history(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["trade_date","open","high","low","close","volume"])
    df = frame.copy()
    aliases = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    for target, candidates in {
        "trade_date": ("trade_date","date","datetime","timestamp"),
        "open": ("open","o"), "high": ("high","h"),
        "low": ("low","l"), "close": ("close","adj close","adj_close","price","c"),
        "volume": ("volume","vol","v"),
    }.items():
        for c in candidates:
            if c in aliases:
                rename[aliases[c]] = target
                break
    df = df.rename(columns=rename)
    if "trade_date" not in df.columns or "close" not in df.columns:
        return pd.DataFrame(columns=["trade_date","open","high","low","close","volume"])
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce", utc=True).dt.tz_convert(None)
    for col in ("open","high","low","close","volume"):
        if col not in df.columns:
            df[col] = np.nan if col != "volume" else 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["trade_date","close"])
    df["high"] = df["high"].fillna(df["close"])
    df["low"]  = df["low"].fillna(df["close"])
    df["volume"] = df["volume"].fillna(0.0)
    df = df[np.isfinite(df["close"]) & (df["close"] > 0) & (df["volume"] >= 0)]
    valid = (df["high"].isna() | (df["high"] >= df["close"])) & (df["low"].isna() | (df["low"] <= df["close"]))
    df = df[valid]
    return (df.sort_values("trade_date").drop_duplicates("trade_date", keep="last").reset_index(drop=True))


def fetch_history(symbol: str, days: int = FETCH_LOOKBACK_DAYS) -> pd.DataFrame:
    frames = []
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT trade_date, open, high, low, close, volume
                FROM market_prices_eod WHERE symbol = :s
                ORDER BY trade_date DESC LIMIT :lim
            """), {"s": symbol, "lim": days}).fetchall()
        if rows:
            frames.append(pd.DataFrame([dict(r._mapping) for r in rows]))
    except Exception as e:
        print(f"⚠️ DB fetch failed for {symbol}: {e}")

    db_df = _normalise_history(frames[0]) if frames else _normalise_history(pd.DataFrame())
    if len(db_df) < MIN_HISTORY_DAYS and HAS_PSXDATA and psxdata is not None:
        try:
            start = (datetime.today() - timedelta(days=days*2)).strftime("%Y-%m-%d")
            end = datetime.today().strftime("%Y-%m-%d")
            provider_df = _quiet_psxdata(psxdata.stocks, symbol, start=start, end=end)
            provider_df = _normalise_history(provider_df)
            if not provider_df.empty:
                frames.append(provider_df)
        except Exception as e:
            print(f"⚠️ psxdata fetch failed for {symbol}: {e}")

    if not frames:
        return db_df
    return _normalise_history(pd.concat(frames, ignore_index=True))


# ============================================================
# compute_metrics  (unchanged — full body identical to your file)
# ============================================================
def compute_metrics(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    if len(df) < MIN_HISTORY_DAYS:
        return None

    close  = df["close"].to_numpy(dtype=float)
    high   = df["high"].to_numpy(dtype=float) if "high" in df else close
    low    = df["low"].to_numpy(dtype=float) if "low" in df else close
    volume = df["volume"].to_numpy(dtype=float) if "volume" in df else np.zeros(len(close))
    last   = float(close[-1])
    if last <= 0 or np.isnan(last):
        return None

    sma20  = float(pd.Series(close).rolling(20).mean().iloc[-1])
    sma50  = float(pd.Series(close).rolling(50).mean().iloc[-1])
    sma200 = float(pd.Series(close).rolling(200).mean().iloc[-1])
    sma50_prev20 = float(pd.Series(close).rolling(50).mean().iloc[-21])
    sma50_slope_pct = (sma50 / sma50_prev20 - 1) * 100 if sma50_prev20 > 0 else 0.0

    w52 = min(len(close), 252)
    high_52w = float(np.max(high[-w52:]))
    low_52w  = float(np.min(low[-w52:]))
    pct_off_high  = (last / high_52w - 1) * 100 if high_52w > 0 else 0.0
    pct_above_low = (last / low_52w - 1) * 100 if low_52w > 0 else 0.0

    def ret(n):
        return float((close[-1] - close[-n-1]) / close[-n-1] * 100) if len(close) > n and close[-n-1] > 0 else 0.0

    ret_1m, ret_3m, ret_6m = ret(21), ret(63), ret(126)

    deltas = pd.Series(close).diff().dropna()
    gains = deltas.clip(lower=0); losses = -deltas.clip(upper=0)
    avg_gain = float(gains.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1])
    avg_loss = float(losses.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1])
    if avg_gain == 0 and avg_loss == 0: rsi = 50.0
    elif avg_loss == 0: rsi = 100.0
    elif avg_gain == 0: rsi = 0.0
    else: rsi = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))

    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26; macd_sig = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_sig
    macd_hist_now  = float(macd_hist.iloc[-1])
    macd_hist_prev = float(macd_hist.iloc[-4]) if len(macd_hist) >= 4 else macd_hist_now
    macd_hist_slope = macd_hist_now - macd_hist_prev

    bb_std = float(pd.Series(close).rolling(20).std().iloc[-1])
    bb_upper = sma20 + 2 * bb_std; bb_lower = sma20 - 2 * bb_std
    bb_range = bb_upper - bb_lower
    bb_position = float((last - bb_lower) / bb_range) if bb_range > 0 else 0.5

    high_s, low_s, close_s = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([high_s - low_s, (high_s - close_s.shift()).abs(), (low_s - close_s.shift()).abs()], axis=1).max(axis=1)
    atr = float(tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean().iloc[-1])

    daily_ret = pd.Series(close).pct_change().dropna()
    vol_ann = float(daily_ret.std() * np.sqrt(252))
    ann_ret = float(daily_ret.mean() * 252)
    sharpe  = (ann_ret - RISK_FREE_RATE_ANNUAL) / vol_ann if vol_ann > 0 else 0.0

    recent = close[-126:] if len(close) >= 126 else close
    peak = np.maximum.accumulate(recent)
    dd = (recent - peak) / peak
    max_dd = float(dd.min())

    avg_vol_20d = float(np.mean(volume[-20:])) if len(volume) >= 20 else 0.0
    avg_vol_90d = float(np.mean(volume[-90:])) if len(volume) >= 90 else avg_vol_20d
    vol_ratio_20_90 = float(avg_vol_20d / avg_vol_90d) if avg_vol_90d > 0 else 1.0
    turnover = pd.Series(close * np.nan_to_num(volume, nan=0.0))
    daily_pkr = float(turnover.iloc[-20:].mean()) if len(turnover) >= 20 else 0.0

    recent_low_20 = float(np.min(low[-20:]))
    support_candidates = [recent_low_20, sma20, sma50]
    valid_supports = [s for s in support_candidates if s < last and s > 0]
    support = max(valid_supports) if valid_supports else last * 0.92
    stop_loss = support - 0.5 * atr

    risk_per_share = last - stop_loss
    if high_52w > last * 1.02 and high_52w <= last * 1.35:
        target = high_52w
    else:
        target = last + 2.0 * risk_per_share
    reward_per_share = target - last
    rr_ratio = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0.0

    return {
        "last_close": round(last, 2),
        "sma20": round(sma20, 2), "sma50": round(sma50, 2), "sma200": round(sma200, 2),
        "price_vs_sma20_pct": round((last/sma20 - 1)*100, 2) if sma20 > 0 else 0.0,
        "price_vs_sma50_pct": round((last/sma50 - 1)*100, 2) if sma50 > 0 else 0.0,
        "price_vs_sma200_pct": round((last/sma200 - 1)*100, 2) if sma200 > 0 else 0.0,
        "sma50_vs_sma200_pct": round((sma50/sma200 - 1)*100, 2) if sma200 > 0 else 0.0,
        "sma50_slope_20d_pct": round(sma50_slope_pct, 2),
        "high_52w": round(high_52w, 2), "low_52w": round(low_52w, 2),
        "pct_off_52w_high": round(pct_off_high, 2),
        "pct_above_52w_low": round(pct_above_low, 2),
        "return_1m_pct": round(ret_1m, 2), "return_3m_pct": round(ret_3m, 2), "return_6m_pct": round(ret_6m, 2),
        "rsi_14": round(rsi, 1),
        "macd_hist": round(macd_hist_now, 3), "macd_hist_slope_3d": round(macd_hist_slope, 3),
        "bb_position": round(bb_position, 2),
        "volatility_ann_pct": round(vol_ann*100, 2), "sharpe_ann": round(sharpe, 2),
        "max_drawdown_pct": round(max_dd*100, 2), "atr_14": round(atr, 2),
        "avg_daily_volume_pkr": round(daily_pkr, 0),
        "volume_ratio_20d_90d": round(vol_ratio_20_90, 2),
        "history_days": len(df),
        "latest_trade_date": pd.Timestamp(df["trade_date"].iloc[-1]).date().isoformat() if "trade_date" in df.columns else None,
        "data_age_days": max(0, (datetime.now().date() - pd.Timestamp(df["trade_date"].iloc[-1]).date()).days) if "trade_date" in df.columns else 9999,
        "entry_price": round(last, 2), "stop_loss": round(stop_loss, 2),
        "target_price": round(target, 2), "rr_ratio": round(rr_ratio, 2),
        "risk_per_share_pct": round((risk_per_share/last)*100, 2) if last > 0 else 0.0,
    }


# ============================================================
# Filters + scoring  (unchanged)
# ============================================================
def apply_today_filters(m: Dict[str, Any]) -> Optional[str]:
    if m.get("latest_trade_date") is None or m.get("data_age_days", 9999) > MAX_DATA_AGE_CALENDAR_DAYS:
        return "stale_data"
    if m["last_close"] < FILTER_MIN_PRICE: return "penny_stock"
    if m["avg_daily_volume_pkr"] < FILTER_MIN_AVG_DAILY_PKR: return "illiquid"
    if m["rsi_14"] > FILTER_MAX_RSI: return "overbought"
    if m["rsi_14"] < FILTER_MIN_RSI: return "oversold_broken"
    if m["price_vs_sma20_pct"] > FILTER_MAX_PCT_ABOVE_SMA20: return "extended_above_sma20"
    if m["pct_off_52w_high"] < -FILTER_MAX_PCT_BELOW_52W_HIGH: return "broken_trend_below_52w"
    if m["volatility_ann_pct"] > FILTER_MAX_VOL_ANN_PCT: return "too_volatile"
    if m["rr_ratio"] < FILTER_MIN_RR_RATIO: return "poor_risk_reward"
    if m["price_vs_sma50_pct"] <= 0: return "below_sma50"
    if m["sma50_vs_sma200_pct"] <= 0: return "sma50_below_sma200"
    return None


def _pct_rank(s: pd.Series) -> pd.Series:
    return s.rank(pct=True).fillna(0.0) * 100


def _score_rows(rows: List[Dict[str, Any]], profile: str) -> List[Dict[str, Any]]:
    df = pd.DataFrame(rows); w = PROFILES[profile]
    df["_pct_from_sma20"] = df["price_vs_sma20_pct"].abs()
    df["_pct_from_sma50"] = df["price_vs_sma50_pct"].abs()
    df["_rsi_quality"] = df["rsi_14"].apply(lambda r: max(0.0, 100.0 - abs(r - 55) * 2.0))
    df["_bb_quality"]  = df["bb_position"].apply(lambda p: max(0.0, min(100.0, 100.0 * (1 - abs(p - 0.5) * 2))))
    df["score_entry_positioning"] = (
        _pct_rank(-df["_pct_from_sma20"]) + _pct_rank(-df["_pct_from_sma50"])
        + df["_rsi_quality"] + df["_bb_quality"]
    ) / 4.0

    df["_upside_to_52w"] = df["pct_off_52w_high"].apply(lambda x: 100.0 if x >= 0 else max(0.0, 100.0 + x))
    df["score_risk_reward"] = (
        _pct_rank(df["rr_ratio"]) + _pct_rank(-df["risk_per_share_pct"])
        + _pct_rank(df["_upside_to_52w"]) + _pct_rank(df["sharpe_ann"])
    ) / 4.0

    def momentum_quality(r1m, r3m, macd_slope):
        if r1m <= 0: m1 = 0.0
        elif r1m > 20: m1 = max(0.0, 40.0 + 60.0 * (1 - (r1m - 20)/30))
        else: m1 = 100.0 * (1 - abs(r1m - 8)/12)
        m3 = 100.0 if 5 < r3m < 30 else (60.0 if r3m >= 30 else max(0.0, min(100.0, r3m*3)))
        ms = 100.0 if macd_slope > 0 else 30.0
        return (max(0.0, m1) + max(0.0, m3) + ms) / 3.0

    df["score_momentum_health"] = df.apply(
        lambda r: momentum_quality(r["return_1m_pct"], r["return_3m_pct"], r["macd_hist_slope_3d"]), axis=1)
    df["_t1"] = (df["price_vs_sma50_pct"] > 0).astype(int)
    df["_t2"] = (df["sma50_vs_sma200_pct"] > 0).astype(int)
    df["_t3"] = (df["sma50_slope_20d_pct"] > 0).astype(int)
    df["_t4"] = (df["macd_hist"] > 0).astype(int)
    df["score_trend_structure"] = (df["_t1"] + df["_t2"] + df["_t3"] + df["_t4"]) / 4 * 100
    df["score_liquidity"] = _pct_rank(df["avg_daily_volume_pkr"])
    df["score_volume_confirmation"] = _pct_rank(df["volume_ratio_20d_90d"])
    df["composite_score"] = (
        df["score_entry_positioning"]   * w["entry_positioning"] +
        df["score_risk_reward"]         * w["risk_reward"] +
        df["score_momentum_health"]     * w["momentum_health"] +
        df["score_trend_structure"]     * w["trend_structure"] +
        df["score_liquidity"]           * w["liquidity"] +
        df["score_volume_confirmation"] * w["volume_confirmation"]
    )
    for c in ["score_entry_positioning","score_risk_reward","score_momentum_health",
              "score_trend_structure","score_liquidity","score_volume_confirmation","composite_score"]:
        df[c] = df[c].round(1)
    df = df.sort_values("composite_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1
    df["grade"]  = df.apply(_grade_row, axis=1)
    df["action"] = df.apply(_suggest_action, axis=1)
    df["tags"]   = df.apply(_generate_tags, axis=1)
    drop = ["_pct_from_sma20","_pct_from_sma50","_rsi_quality","_bb_quality",
            "_upside_to_52w","_t1","_t2","_t3","_t4"]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    records = df.to_dict(orient="records")
    for r in records: r["verdict"] = "BUY"
    return records


def _grade_row(row) -> str:
    s, rr = row["composite_score"], row["rr_ratio"]
    if s >= 70 and rr >= 2.0: return "A"
    if s >= 55 and rr >= 1.7: return "B"
    return "C"


def _suggest_action(row) -> str:
    cur, sma20 = row["last_close"], row["sma20"]
    if row["price_vs_sma20_pct"] <= 2.0:
        return f"Watchlist candidate near SMA20 (reference close {cur:.2f}); verify live quote and execution costs"
    pullback = round(sma20 * 1.01, 2)
    if row["price_vs_sma20_pct"] <= 6.0:
        return f"Monitor near {pullback:.2f} (SMA20 reference); confirm price, liquidity and risk before acting"
    return f"Extended versus SMA20; monitor for a pullback near {pullback:.2f} and reassess"


def _generate_tags(row) -> List[str]:
    t = []
    if row["rr_ratio"] >= 2.5: t.append(f"Excellent R:R ({row['rr_ratio']:.1f}:1)")
    elif row["rr_ratio"] >= 2.0: t.append(f"Good R:R ({row['rr_ratio']:.1f}:1)")
    if abs(row["price_vs_sma20_pct"]) <= 3: t.append("Near SMA20 support")
    if row["rsi_14"] < 45: t.append("RSI cooled")
    elif 45 <= row["rsi_14"] <= 65: t.append("RSI healthy")
    if row["macd_hist"] > 0 and row["macd_hist_slope_3d"] > 0: t.append("MACD rising")
    if row["volume_ratio_20d_90d"] > 1.3: t.append("Volume uptick")
    if row["pct_off_52w_high"] >= -5: t.append("Near 52w high")
    elif -30 <= row["pct_off_52w_high"] <= -15: t.append("Pullback in uptrend")
    if row["sharpe_ann"] >= 1.0: t.append("Good risk-adjusted return")
    if row["max_drawdown_pct"] > -10: t.append("Shallow drawdown")
    if row["avg_daily_volume_pkr"] >= 50_000_000: t.append("Highly liquid")
    return t


# ============================================================
# Public entry — full universe
# ============================================================
def _get_metrics_map(universe: List[str], force_refresh: bool) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    CHANGED — metrics now come from the DB snapshot first, not from the network.

    Order of preference:
      1. in-process cache (fastest, if it matches the universe)
      2. quant_metrics_snapshot table in Postgres (written by db.py at 17:45)
      3. compute directly from market_prices_eod (no network)
      4. psxdata fallback (slow — only if everything else failed)
    """
    universe_key = tuple(sorted(universe))

    if (not force_refresh and _METRICS_CACHE["data"]
            and _METRICS_CACHE.get("universe_key") == universe_key):
        print(f"♻️ In-process metrics cache hit ({len(_METRICS_CACHE['data'])} symbols)")
        return _METRICS_CACHE["data"]

    # 1) Snapshot table
    snap = load_latest_quant_snapshot()
    if snap:
        # Restrict to requested universe
        metrics_map = {sym: snap.get(sym) for sym in universe}
        _METRICS_CACHE["ts"] = time.time()
        _METRICS_CACHE["data"] = metrics_map
        _METRICS_CACHE["universe_key"] = universe_key
        print(f"♻️ Snapshot cache hit ({len(snap)} symbols in DB)")
        return metrics_map

    # 2) Compute from Postgres EOD (no network)
    print("⚠️ No snapshot found — computing from market_prices_eod (no network)")
    from db import _compute_metrics_from_db
    metrics_map = _compute_metrics_from_db(universe)
    _METRICS_CACHE["ts"] = time.time()
    _METRICS_CACHE["data"] = metrics_map
    _METRICS_CACHE["universe_key"] = universe_key
    return metrics_map


def run_quant_screener(
    top_n: int = DEFAULT_TOP_N,
    profile: str = "balanced",
    symbol_whitelist: Optional[List[str]] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    profile = profile if profile in PROFILES else "balanced"
    universe = [s.upper() for s in (symbol_whitelist or get_universe())]

    top_n = max(1, min(int(top_n), 100))
    universe_key = tuple(sorted(universe))
    result_key = f"{profile}|{top_n}|{hash(universe_key)}"
    cached_result = _RESULT_CACHE.get(result_key)
    if (not force_refresh and cached_result
            and (time.time() - cached_result["ts"]) < _RESULT_CACHE_TTL_SECONDS):
        print(f"♻️ Result cache hit: {result_key}")
        return cached_result["result"]

    metrics_map = _get_metrics_map(universe, force_refresh)

    passed: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    skipped_stats = {
        "no_data": 0, "too_short": 0, "stale_data": 0,
        "penny_stock": 0, "illiquid": 0, "overbought": 0, "oversold_broken": 0,
        "extended_above_sma20": 0, "broken_trend_below_52w": 0,
        "too_volatile": 0, "poor_risk_reward": 0,
        "below_sma50": 0, "sma50_below_sma200": 0,
    }

    for sym in universe:
        m = metrics_map.get(sym)
        if m is None:
            skipped_stats["no_data"] += 1
            continue
        reason = apply_today_filters(m)
        if reason:
            skipped_stats[reason] = skipped_stats.get(reason, 0) + 1
            if len(rejected) < 40:
                rejected.append({"symbol": sym, "reason": reason,
                                 "metric": _reason_metric_value(reason, m)})
            continue
        passed.append({"symbol": sym, **m})

    print(f"✅ {len(passed)} passed all filters")
    ranked = _score_rows(passed, profile) if passed else []

    result = {
        "as_of": max((m.get("latest_trade_date") or "" for m in metrics_map.values() if m), default=None),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_freshness_limit_days": MAX_DATA_AGE_CALENDAR_DAYS,
        "methodology": "Technical price/volume screening only; not fundamental valuation, a forecast, or a personalized buy recommendation.",
        "profile": profile,
        "top_n": top_n,
        "screened_universe_size": len(universe),
        "candidates": len(passed),
        "top_picks": ranked[:top_n],
        "skipped_breakdown": skipped_stats,
        "rejected_sample": rejected,
    }
    _result_cache_put(result_key, result)
    return result


def _reason_metric_value(reason: str, m: Dict[str, Any]) -> Any:
    table = {
        "penny_stock": ("last_close","PKR"),
        "illiquid": ("avg_daily_volume_pkr","PKR/day"),
        "overbought": ("rsi_14","RSI"),
        "oversold_broken": ("rsi_14","RSI"),
        "extended_above_sma20": ("price_vs_sma20_pct","% above SMA20"),
        "broken_trend_below_52w": ("pct_off_52w_high","% from 52w high"),
        "too_volatile": ("volatility_ann_pct","% annual vol"),
        "poor_risk_reward": ("rr_ratio","R:R"),
        "below_sma50": ("price_vs_sma50_pct","% vs SMA50"),
        "sma50_below_sma200": ("sma50_vs_sma200_pct","SMA50 vs SMA200 %"),
        "stale_data": ("data_age_days","calendar days since latest EOD row"),
    }
    key, label = table.get(reason, (None, ""))
    if key is None: return None
    return {"value": m.get(key), "unit": label}


# ============================================================
# Public entry — specific symbols
# ============================================================
def screen_specific_symbols(symbols: List[str], profile: str = "balanced") -> Dict[str, Any]:
    return run_quant_screener(top_n=len(symbols), profile=profile,
                              symbol_whitelist=symbols, force_refresh=True)


# ============================================================
# Manual refresh — delegated to db.py
# ============================================================
def refresh_universe_prices(max_symbols: Optional[int] = None) -> Dict[str, Any]:
    """Manual trigger — writes a fresh snapshot to Postgres."""
    if max_symbols:
        # only refresh a slice
        universe = get_universe()[:max_symbols]
        _universe_override = universe
    else:
        _universe_override = None
    result = refresh_quant_snapshot(fetch_prices=True)
    _RESULT_CACHE.clear()
    _METRICS_CACHE.update({"ts": 0.0, "data": {}, "universe_key": None})
    return result


# ============================================================
# Auto-start  (CHANGED — refresher now lives in db.py)
# ============================================================
if os.getenv("PSX_QUANT_AUTO_START", "true").lower() in {"1", "true", "yes"}:
    try:
        init_db()                       # make sure tables exist
        _db_start_daily_refresher()     # 17:45 daily job (idempotent)
    except Exception as _e:
        print(f"⚠️ Quant auto-start skipped: {_e}")


# ============================================================
# CLI  (unchanged)
# ============================================================
if __name__ == "__main__":
    import sys, json
    arg1 = sys.argv[1] if len(sys.argv) > 1 else "balanced"
    arg2 = sys.argv[2] if len(sys.argv) > 2 else "10"
    if arg1 == "refresh":
        print(json.dumps(refresh_universe_prices(), indent=2))
    else:
        profile, n = arg1, int(arg2)
        result = run_quant_screener(top_n=n, profile=profile)
        print(f"\n{'='*88}")
        print(f"BUY-TODAY SCREEN · profile={profile} · {result['candidates']}/{result['screened_universe_size']} passed")
        print(f"{'='*88}\n")
        for p in result["top_picks"]:
            print(f"#{p['rank']:>2} [{p['grade']}]  {p['symbol']:<6}  "
                  f"score={p['composite_score']:>5.1f}  "
                  f"entry={p['entry_price']:>8.2f}  stop={p['stop_loss']:>8.2f}  "
                  f"target={p['target_price']:>8.2f}  R:R={p['rr_ratio']:>4.1f}:1")
            print(f"     {p['action']}")
            if p["tags"]:
                print(f"     ✓ {' · '.join(p['tags'])}")
            print()
        if result["skipped_breakdown"]:
            print(f"\n🚫 Rejections: {result['skipped_breakdown']}")