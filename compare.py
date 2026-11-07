# compare.py
"""
Comparative analysis — ONE verdict engine.

Verdicts (BUY/SELL/HOLD) come from agents_graph.build_psx_graph().invoke()
which is the SINGLE SOURCE OF TRUTH for all signals. This module computes
technicals + prose; it does NOT decide signals itself.

Why: previously compare.py and agents_graph.py ran two different LLM
pipelines and produced two contradictory answers for the same stock.
Now they cannot disagree — compare.py reads what agents_graph.py produced.
"""

import re
import threading
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sqlalchemy import text

from db import engine, load_psx_market_prices
from llm import generate_text, LLMUnavailable

logger = logging.getLogger(__name__)


# ── LightGBM (soft dependency) ──
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    lgb = None
    HAS_LIGHTGBM = False
    print("ℹ️ lightgbm not installed — forecast will be skipped. pip install lightgbm")


HISTORY_DAYS = 90
RAW_FETCH_LIMIT = max(HISTORY_DAYS * 2, 180)
LEAD_LAG_MIN_CORR = 0.20

# ── LightGBM forecast cache ──
_FORECAST_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_FORECAST_CACHE_TTL = 900
_FORECAST_LOCK = threading.Lock()

# ── Canonical verdict cache (agents_graph is expensive) ──
_VERDICT_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_VERDICT_CACHE_TTL = 1800  # 30 min — one graph run per (stock, date) per half-hour
_VERDICT_LOCK = threading.Lock()

# ── Graph singleton ──
_GRAPH = None
_GRAPH_LOCK = threading.Lock()


# ═══════════════════════════════════════════════════════════════════
# SINGLE SOURCE OF TRUTH — verdict comes from agents_graph.py
# ═══════════════════════════════════════════════════════════════════
def _get_graph():
    """Lazy singleton — the graph object is expensive to construct."""
    global _GRAPH
    if _GRAPH is None:
        with _GRAPH_LOCK:
            if _GRAPH is None:
                from agents_graph import build_psx_graph
                _GRAPH = build_psx_graph()
    return _GRAPH


def _extract_verdict_block(report_text: str) -> Dict[str, Any]:
    """Same regex used in app.py:_extract_signal_fields — kept local."""
    if not report_text:
        return {"verdict": "HOLD", "confidence": 0.0, "holding_days": None}

    verdict = "HOLD"
    vm = re.search(r"VERDICT\s*[:\-]\s*(BUY|SELL|HOLD)\b", report_text, re.IGNORECASE)
    if vm:
        verdict = vm.group(1).upper()

    confidence = 0.0
    cm = re.search(r"CONFIDENCE\s*[:\-]\s*(0?\.\d+|[01](?:\.\d+)?)", report_text, re.IGNORECASE)
    if cm:
        try:
            confidence = min(1.0, max(0.0, float(cm.group(1))))
        except ValueError:
            pass

    holding_days = None
    hm = re.search(r"HOLDING[_\s]*PERIOD\s*[:\-]\s*(\d{1,3})", report_text, re.IGNORECASE)
    if hm:
        try:
            holding_days = int(hm.group(1))
        except ValueError:
            pass

    return {"verdict": verdict, "confidence": confidence, "holding_days": holding_days}


def get_verdict_for_symbol(symbol: str, as_of: str) -> Dict[str, Any]:
    """
    Run agents_graph.build_psx_graph().invoke() and return the verdict.

    Cached per (symbol, as_of) for 30 minutes — one graph run per stock per
    half hour. Multiple stock rows re-uses the same graph instance.
    """
    cache_key = f"{symbol}:{as_of}"

    with _VERDICT_LOCK:
        cached = _VERDICT_CACHE.get(cache_key)
        if cached and time.time() - cached[0] < _VERDICT_CACHE_TTL:
            return dict(cached[1])

    try:
        graph = _get_graph()
        state = {
            "query": f"Analyze {symbol} as of {as_of}",
            "symbol": symbol,
            "numerical_data": {},
            "retrieved_docs": [],
            "sentiment_summary": "",
            "risk_alerts": [],
            "final_report": "",
        }
        final = graph.invoke(state)
        report = final.get("final_report", "") or ""
        v = _extract_verdict_block(report)
        result = {
            "symbol": symbol,
            "verdict": v["verdict"],
            "holding_days": v["holding_days"],
            "confidence": v["confidence"],
            "source": "agents_graph",
        }
    except Exception as exc:
        logger.exception("agents_graph.invoke() failed for %s: %s", symbol, exc)
        result = {
            "symbol": symbol,
            "verdict": "HOLD",
            "holding_days": None,
            "confidence": 0.0,
            "source": "fallback",
            "error": str(exc)[:200],
        }

    with _VERDICT_LOCK:
        _VERDICT_CACHE[cache_key] = (time.time(), result)
        if len(_VERDICT_CACHE) > 64:
            oldest = min(_VERDICT_CACHE, key=lambda k: _VERDICT_CACHE[k][0])
            if oldest != cache_key:
                _VERDICT_CACHE.pop(oldest, None)

    return result


# ═══════════════════════════════════════════════════════════════════
# LightGBM 7-day forecaster (per-symbol cached)
# ═══════════════════════════════════════════════════════════════════
def safe_corrcoef(x, y):
    if len(x) < 2 or len(y) < 2:
        return 0.0
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def _lightgbm_forecast(df: pd.DataFrame, periods: int = 7, symbol: str = "") -> List[Dict[str, Any]]:
    if not HAS_LIGHTGBM or df is None or len(df) < 60:
        return []

    cache_key = f"{symbol}:{periods}" if symbol else ""
    if cache_key:
        with _FORECAST_LOCK:
            cached = _FORECAST_CACHE.get(cache_key)
            if cached and time.time() - cached[0] < _FORECAST_CACHE_TTL:
                return list(cached[1])

    try:
        feat = df.copy().sort_values("trade_date").reset_index(drop=True)
        feat["dow"]   = feat["trade_date"].dt.dayofweek
        feat["dom"]   = feat["trade_date"].dt.day
        feat["month"] = feat["trade_date"].dt.month

        for lag in (1, 2, 3, 5, 10, 20):
            feat[f"lag_{lag}"] = feat["close"].shift(lag)
        for win in (5, 10, 20):
            feat[f"roll_mean_{win}"] = feat["close"].rolling(win).mean()
            feat[f"roll_std_{win}"]  = feat["close"].rolling(win).std()
        feat["ret_1"] = feat["close"].pct_change(1)
        feat["ret_5"] = feat["close"].pct_change(5)
        feat = feat.dropna().reset_index(drop=True)
        if len(feat) < 40:
            return []

        feature_cols = [c for c in feat.columns
                        if c not in ("trade_date", "open", "high", "low", "close", "volume")]
        X, y = feat[feature_cols].values, feat["close"].values

        model = lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.05, max_depth=5,
            num_leaves=15, min_child_samples=10, random_state=42, verbose=-1,
        )
        model.fit(X, y)

        last_date = pd.Timestamp(feat["trade_date"].iloc[-1])
        close_history = list(feat["close"].values)
        base = feat.iloc[-1][feature_cols].copy()

        forecast: List[Dict[str, Any]] = []
        for i in range(periods):
            nxt = last_date + pd.offsets.BDay(i + 1)
            f = base.copy()
            f["dow"], f["dom"], f["month"] = nxt.dayofweek, nxt.day, nxt.month
            for lag in (1, 2, 3, 5, 10, 20):
                if len(close_history) > lag:
                    f[f"lag_{lag}"] = close_history[-lag]
            for win in (5, 10, 20):
                if len(close_history) >= win:
                    arr = np.array(close_history[-win:])
                    f[f"roll_mean_{win}"] = arr.mean()
                    f[f"roll_std_{win}"]  = arr.std()
            if len(close_history) >= 2 and close_history[-2] > 0:
                f["ret_1"] = (close_history[-1] - close_history[-2]) / close_history[-2]
            if len(close_history) >= 6 and close_history[-6] > 0:
                f["ret_5"] = (close_history[-1] - close_history[-6]) / close_history[-6]

            pred = float(model.predict(f.values.reshape(1, -1))[0])
            if not np.isfinite(pred) or pred <= 0:
                pred = float(close_history[-1])
            pred = max(0.1, pred)

            forecast.append({
                "trade_date": nxt.strftime("%Y-%m-%d"),
                "predicted_close": round(pred, 2),
            })
            close_history.append(pred)

        if cache_key:
            with _FORECAST_LOCK:
                _FORECAST_CACHE[cache_key] = (time.time(), forecast)
                if len(_FORECAST_CACHE) > 64:
                    oldest = min(_FORECAST_CACHE, key=lambda k: _FORECAST_CACHE[k][0])
                    if oldest != cache_key:
                        _FORECAST_CACHE.pop(oldest, None)

        return forecast
    except Exception as e:
        logger.warning("LightGBM forecast failed for %s: %s", symbol or "?", e)
        return []


# ═══════════════════════════════════════════════════════════════════
# Data + indicators
# ═══════════════════════════════════════════════════════════════════
def _read_price_history(symbol: str, limit: int) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT trade_date, open, high, low, close, volume
            FROM market_prices_eod
            WHERE symbol = :s AND close IS NOT NULL
            ORDER BY trade_date DESC LIMIT :lim
        """), {"s": symbol, "lim": limit}).fetchall()
    return [dict(r._mapping) for r in rows]


def fetch_and_calculate_metrics(symbols: List[str]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    per_symbol: Dict[str, Any] = {}
    history: Dict[str, Any] = {}

    for sym in symbols:
        rows = _read_price_history(sym, RAW_FETCH_LIMIT)

        if len(rows) < 20:
            print(f"🔄 Auto-fetching data for {sym}...")
            try:
                load_psx_market_prices(target_symbol=sym)
            except Exception as e:
                print(f"⚠️ Failed to auto-fetch {sym}: {e}")
            rows = _read_price_history(sym, RAW_FETCH_LIMIT)

        if len(rows) < 20:
            per_symbol[sym] = {"data_available": False}
            continue

        df = pd.DataFrame(rows).sort_values("trade_date")
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        for col in ["close", "volume", "open", "high", "low"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["close", "volume", "open", "high", "low"]).reset_index(drop=True)
        if len(df) < 20:
            per_symbol[sym] = {"data_available": False}
            continue

        delta = df["close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        df["rsi"] = 100 - (100 / (1 + (gain / loss)))

        exp12 = df["close"].ewm(span=12, adjust=False).mean()
        exp26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = exp12 - exp26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        df["bb_mid"]   = df["close"].rolling(window=20).mean()
        df["bb_std"]   = df["close"].rolling(window=20).std()
        df["bb_upper"] = df["bb_mid"] + (df["bb_std"] * 2)
        df["bb_lower"] = df["bb_mid"] - (df["bb_std"] * 2)

        avg_vol_30d = df["volume"].tail(30).mean()
        last, prev = df.iloc[-1], df.iloc[-2]
        last_close = float(last["close"])
        prev_close = float(prev["close"])
        change_pct = ((last_close - prev_close) / prev_close) * 100 if prev_close > 0 else 0
        vol_ratio = float(last["volume"]) / float(avg_vol_30d) if avg_vol_30d > 0 else 0

        forecast_data = _lightgbm_forecast(df, periods=7, symbol=sym)

        per_symbol[sym] = {
            "data_available": True,
            "close": round(last_close, 2),
            "change_pct": round(change_pct, 2),
            "open": round(float(last["open"]), 2),
            "high": round(float(last["high"]), 2),
            "low": round(float(last["low"]), 2),
            "volume": int(last["volume"]),
            "avg_volume_30d": int(avg_vol_30d) if not pd.isna(avg_vol_30d) else 0,
            "vol_ratio": round(vol_ratio, 2),
            "rsi": round(float(df["rsi"].iloc[-1]), 1) if not pd.isna(df["rsi"].iloc[-1]) else 50.0,
            "macd": round(float(df["macd"].iloc[-1]), 2) if not pd.isna(df["macd"].iloc[-1]) else 0.0,
            "bb_mid": round(float(df["bb_mid"].iloc[-1]), 2) if not pd.isna(df["bb_mid"].iloc[-1]) else round(last_close, 2),
        }

        chart_df = df.tail(HISTORY_DAYS).reset_index(drop=True)
        history[sym] = {
            "dates": [row["trade_date"].strftime("%Y-%m-%d") for _, row in chart_df.iterrows()],
            "close": [round(float(row["close"]), 2) for _, row in chart_df.iterrows()],
            "rsi": [round(float(row["rsi"]), 1) if not pd.isna(row["rsi"]) else 50.0 for _, row in chart_df.iterrows()],
            "macd": [round(float(row["macd"]), 2) if not pd.isna(row["macd"]) else 0.0 for _, row in chart_df.iterrows()],
            "macd_signal": [round(float(row["macd_signal"]), 2) if not pd.isna(row["macd_signal"]) else 0.0 for _, row in chart_df.iterrows()],
            "macd_hist": [round(float(row["macd_hist"]), 2) if not pd.isna(row["macd_hist"]) else 0.0 for _, row in chart_df.iterrows()],
            "bb_upper": [round(float(row["bb_upper"]), 2) if not pd.isna(row["bb_upper"]) else None for _, row in chart_df.iterrows()],
            "bb_lower": [round(float(row["bb_lower"]), 2) if not pd.isna(row["bb_lower"]) else None for _, row in chart_df.iterrows()],
            "forecast": forecast_data,
        }

    return per_symbol, history


def calculate_lead_lag(symbols: List[str], max_lag: int = 3) -> Dict[str, Any]:
    prices: Dict[str, List[float]] = {}
    for sym in symbols:
        with engine.connect() as conn:
            rows = conn.execute(text(
                "SELECT close FROM market_prices_eod "
                "WHERE symbol = :s AND close IS NOT NULL "
                "ORDER BY trade_date DESC LIMIT 40"
            ), {"s": sym}).fetchall()
        prices[sym] = []
        for r in reversed(rows):
            if r[0] is not None:
                try:
                    val = float(r[0])
                    if not np.isnan(val) and val > 0:
                        prices[sym].append(val)
                except (ValueError, TypeError):
                    pass

    valid_symbols = [s for s in symbols if s in prices and len(prices[s]) >= 20]
    if len(valid_symbols) < 2:
        return {"lags": [], "pairs": {}}

    min_len = min(len(prices[s]) for s in valid_symbols)
    returns = {sym: np.diff(prices[sym][-min_len:]) / prices[sym][-min_len:-1]
               for sym in valid_symbols}

    lags = list(range(-max_lag, max_lag + 1))
    lead_lag_data = {f"{s1}-{s2}": [] for i, s1 in enumerate(valid_symbols) for s2 in valid_symbols[i + 1:]}

    for pair in lead_lag_data.keys():
        s1, s2 = pair.split("-")
        r1, r2 = returns[s1], returns[s2]
        for lag in lags:
            if lag < 0:
                corr = safe_corrcoef(r1[-lag:], r2[:lag])
            elif lag > 0:
                corr = safe_corrcoef(r1[:-lag], r2[lag:])
            else:
                corr = safe_corrcoef(r1, r2)
            lead_lag_data[pair].append(corr)

    return {"lags": lags, "pairs": lead_lag_data}


def _summarize_lead_lag(lead_lag: Dict[str, Any], threshold: float = LEAD_LAG_MIN_CORR) -> str:
    if not lead_lag or not lead_lag.get("pairs"):
        return "No lead-lag data available (insufficient overlapping history)."

    lags = lead_lag["lags"]
    pairs = lead_lag["pairs"]
    lines: List[str] = []

    for pair_name, corrs in pairs.items():
        if not corrs or len(corrs) != len(lags):
            continue
        zero_idx = lags.index(0) if 0 in lags else None
        zero_corr = corrs[zero_idx] if zero_idx is not None else 0.0
        abs_corrs = [abs(c) for c in corrs]
        max_idx = abs_corrs.index(max(abs_corrs))
        max_corr = corrs[max_idx]
        max_lag = lags[max_idx]
        s1, s2 = pair_name.split("-")

        if abs(max_corr) < threshold:
            verdict = "no meaningful relationship (all lags weak)"
        elif max_lag == 0:
            verdict = f"co-move on the same day (same-day corr {zero_corr:+.2f})"
        elif max_lag < 0:
            verdict = f"{s1} leads {s2} by {abs(max_lag)} day(s) (corr {max_corr:+.2f} at lag {max_lag})"
        else:
            verdict = f"{s2} leads {s1} by {abs(max_lag)} day(s) (corr {max_corr:+.2f} at lag {max_lag})"

        lines.append(
            f"- {s1} vs {s2}: same-day corr {zero_corr:+.2f} · "
            f"strongest {max_corr:+.2f} at lag {max_lag} · {verdict}"
        )

    return "\n".join(lines) if lines else "No lead-lag data available (insufficient overlapping history)."


# ═══════════════════════════════════════════════════════════════════
# Main entry
# ═══════════════════════════════════════════════════════════════════
def run_comparative_analysis(symbols: List[str]) -> Dict[str, Any]:
    as_of = pd.Timestamp.now().strftime("%Y-%m-%d")

    per_symbol, history = fetch_and_calculate_metrics(symbols)
    lead_lag = calculate_lead_lag(symbols)
    lead_lag_summary = _summarize_lead_lag(lead_lag)

    # ══════════════════════════════════════════════════════════
    # CANONICAL VERDICTS — from agents_graph.py only
    # ══════════════════════════════════════════════════════════
    print(f"🔍 Fetching canonical verdicts from agents_graph for {len(symbols)} stock(s)…",
          flush=True)
    t0 = time.time()

    verdict_map: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {pool.submit(get_verdict_for_symbol, sym, as_of): sym for sym in symbols}
        for future in as_completed(futures):
            sym = futures[future]
            try:
                verdict_map[sym] = future.result()
            except Exception as exc:
                logger.exception("Verdict fetch failed for %s: %s", sym, exc)
                verdict_map[sym] = {
                    "symbol": sym, "verdict": "HOLD", "confidence": 0.0,
                    "holding_days": None, "source": "fallback",
                }

    elapsed = time.time() - t0
    print(f"   ✓ Verdicts for {len(symbols)} stocks in {elapsed:.1f}s", flush=True)

    # Preserve original symbol order
    signals = [
        {
            "symbol": sym,
            "verdict": verdict_map[sym]["verdict"],
            "holding_days": verdict_map[sym]["holding_days"],
            "confidence": verdict_map[sym]["confidence"],
            "source": verdict_map[sym].get("source", "agents_graph"),
        }
        for sym in symbols
    ]

    # ══════════════════════════════════════════════════════════
    # Descriptive prose only (no SIGNALS — verdicts already canonical)
    # ══════════════════════════════════════════════════════════
    prompt = f"""
You are a strict, objective technical analyst. Compare these PSX stocks based on the following metrics:
{per_symbol}

LEAD-LAG CORRELATION (daily returns, lags -3 to +3):
{lead_lag_summary}

Write a 4-7 sentence professional financial summary following these exact rules:

RELATIVE STRENGTH: Do not use subjective terms like "resilience" if the stock is below its Bollinger Mid. Use precise language: "posted the highest relative intraday gain" (highest % change) or "showed the least weakness."

LIQUIDITY & VOLUME: Distinguish between 30d Avg Volume and today's Vol Ratio. Flag Vol Ratio < 0.5x as "dead volume/lack of trading interest." Flag Vol Ratio > 2.0x with a flat price (< 0.5% change) as "potential distribution."

RISK: Mention negative MACD and RSI < 40. Note the steepest MACD decay.

LEAD-LAG (CRITICAL):
1. Only discuss lead-lag for pairs where |strongest correlation| >= 0.20. Below that, explicitly say "no meaningful lead-lag relationship was detected."
2. If the strongest correlation is at lag 0, describe the pair as "co-moving on the same day" — do NOT invent a leader.
3. If a clear leader-follower pattern exists (|corr| >= 0.20 at lag ≠ 0), name the leading stock and the lead time in days.
4. Cite the exact correlation numbers from the LEAD-LAG CORRELATION block. Never fabricate lag values.
5. If there are no pairs to discuss, say "lead-lag analysis was inconclusive due to insufficient overlapping data."

SECTORS (CRITICAL): Do NOT hallucinate sector names. Only mention sector diversification if the sector names are explicitly provided in the data payload. If not provided, use generic terms like "these equities."

DO NOT output a SIGNALS block. Verdicts (BUY/SELL/HOLD) are already computed by a separate downstream system and will be merged into this response automatically. Just write the descriptive paragraph.

Then the final line:

Disclaimer: This is for Educational purposes only.

🚫 CRITICAL FORMATTING RULE: Do NOT use Markdown tables anywhere in your response.
"""

    try:
        summary_clean = generate_text(prompt, "You are a professional financial analyst.").strip()
    except LLMUnavailable as exc:
        logger.warning("Compare prose unavailable: %s", exc)
        summary_clean = (
            "Comparative summary unavailable (all LLM providers down). "
            "Per-stock verdicts below come from the canonical agents_graph pipeline."
        )

    return {
        "symbols": symbols,
        "as_of": as_of,
        "history_days": HISTORY_DAYS,
        "per_symbol": per_symbol,
        "history": history,
        "lead_lag": lead_lag,
        "lead_lag_summary": lead_lag_summary,
        "comparative_summary": summary_clean,
        "signals": signals,
    }