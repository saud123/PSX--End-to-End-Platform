"""
backtest.py — Decision viewer for V1 vs V2, Jan–Oct 2026.

No simulation. No trades. No P&L. Just:
    For each stock, for each date: what did V1 and V2 say?

USAGE:
    python backtest.py                                    # defaults
    python backtest.py --dates 2026-02-10,2026-04-15,...  # explicit
    python backtest.py --stocks OGDC,PSO,LUCK,ENGRO
    python backtest.py --random --seed 42
"""

import os
import re
import json
import time
import logging
import argparse
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
from sqlalchemy import text

from db import engine

import agents_graph as _ag
from agents_graph import build_psx_graph as build_v1

try:
    import agents_graph_v2 as _ag2
    from agents_graph_v2 import build_psx_graph as build_v2
    HAS_V2 = True
except ImportError as _imp_err:
    build_v2 = None
    HAS_V2 = False
    print(f"⚠️ V2 not available: {_imp_err}", flush=True)


# ═══════════════════════════════════════════════════════════════════
# RATE LIMITER (unchanged from previous fix)
# ═══════════════════════════════════════════════════════════════════
class SessionLimitReached(BaseException):
    pass


class RateLimiter:
    def __init__(self, rpm: int = 8, daily: int = 400):
        self.rpm = max(1, int(rpm))
        self.daily = max(1, int(daily))
        self._minute_window: List[float] = []
        self._calls_today = 0
        self._total = 0
        self._blocked = 0
        self._day = time.strftime("%Y-%m-%d")
        self._lock = threading.Lock()
        self._last_sleep_log = 0.0
        self._by_label: Dict[str, int] = {}

    def _reset_day_if_needed(self):
        today = time.strftime("%Y-%m-%d")
        if today != self._day:
            self._day = today
            self._calls_today = 0

    def acquire(self, label: str = ""):
        with self._lock:
            self._reset_day_if_needed()
            if self._calls_today >= self.daily:
                raise SessionLimitReached(
                    f"Daily LLM cap reached ({self._calls_today}/{self.daily})."
                )
            now = time.monotonic()
            cutoff = now - 60.0
            self._minute_window = [t for t in self._minute_window if t > cutoff]
            if len(self._minute_window) >= self.rpm:
                sleep_for = self._minute_window[0] - cutoff + 0.1
                if sleep_for > 0:
                    self._blocked += 1
                    if now - self._last_sleep_log > 10:
                        print(f"  ⏳ throttle: sleeping {sleep_for:.1f}s "
                              f"(calls {self._calls_today}/{self.daily})", flush=True)
                        self._last_sleep_log = now
                    time.sleep(sleep_for)
                    now = time.monotonic()
                    cutoff = now - 60.0
                    self._minute_window = [t for t in self._minute_window if t > cutoff]
            self._minute_window.append(now)
            self._calls_today += 1
            self._total += 1
            if label:
                self._by_label[label] = self._by_label.get(label, 0) + 1

    @property
    def calls_used(self) -> int:
        return self._total

    def summary(self) -> str:
        parts = [f"{self._calls_today}/{self.daily} today",
                 f"{self._total} total",
                 f"{self._blocked} sleeps"]
        if self._by_label:
            parts.append("by source: " + ", ".join(f"{k}={v}" for k, v in self._by_label.items()))
        return " | ".join(parts)


_LIMITER: Optional[RateLimiter] = None
_wrapped_ids: set = set()


def _wrap_method_on_instance(obj, attr: str, limiter: RateLimiter, label: str = "") -> bool:
    if obj is None or not hasattr(obj, attr):
        return False
    original = getattr(obj, attr)
    if not callable(original):
        return False
    key = (label, id(obj), attr)
    if key in _wrapped_ids:
        return False

    def wrapped(*a, **kw):
        limiter.acquire(label)
        return original(*a, **kw)
    wrapped._rate_limited = True
    _wrapped_ids.add(key)
    try:
        setattr(obj, attr, wrapped)
        return True
    except (AttributeError, TypeError):
        return False


def install_rate_limiter(limiter: RateLimiter) -> int:
    global _LIMITER
    _LIMITER = limiter
    modules = [(_ag, "agents_graph")]
    if HAS_V2:
        modules.append((_ag2, "agents_graph_v2"))

    wrapped = 0
    for mod, label in modules:
        for attr in ("gemini_client", "llm", "llm_model", "model",
                     "chat_model", "gemini", "llm_instance",
                     "client", "model_client"):
            obj = getattr(mod, attr, None)
            if obj is None:
                continue
            inner = getattr(obj, "models", None)
            if inner is not None:
                if _wrap_method_on_instance(inner, "generate_content", limiter,
                                            f"{label}.{attr}.models.generate_content"):
                    print(f"  ✅ rate-limited  {label}.{attr}.models.generate_content", flush=True)
                    wrapped += 1
            for method in ("invoke", "generate", "generate_content",
                           "_generate", "ainvoke", "agenerate"):
                if _wrap_method_on_instance(obj, method, limiter, f"{label}.{attr}.{method}"):
                    print(f"  ✅ rate-limited  {label}.{attr}.{method}", flush=True)
                    wrapped += 1

        for fn_name in ("generate_text", "call_llm", "run_llm", "ask_llm"):
            key = (label, fn_name)
            if key in _wrapped_ids:
                continue
            fn = getattr(mod, fn_name, None)
            if fn is None or not callable(fn):
                continue

            def make(orig, lim, lbl):
                def w(*a, **kw):
                    lim.acquire(lbl)
                    return orig(*a, **kw)
                w._rate_limited = True
                return w
            try:
                setattr(mod, fn_name, make(fn, limiter, f"{label}.{fn_name}"))
                _wrapped_ids.add(key)
                print(f"  ✅ rate-limited  {label}.{fn_name}", flush=True)
                wrapped += 1
            except (AttributeError, TypeError):
                pass

    if wrapped == 0:
        print("\n⚠️  WARNING: throttle INACTIVE.\n", flush=True)
    else:
        print(f"✅ Rate limiter active on {wrapped} entry point(s)\n", flush=True)
    return wrapped


logger = logging.getLogger("backtest")


# ═══════════════════════════════════════════════════════════════════
# CLOCK FREEZING — no future data
# ═══════════════════════════════════════════════════════════════════
_current_as_of: Optional[str] = None
_original_fetch_market_data = _ag.fetch_market_data


def _patched_fetch_market_data(symbol: str) -> Dict[str, Any]:
    if _current_as_of is None:
        return _original_fetch_market_data(symbol)
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT trade_date, close, volume
            FROM market_prices_eod
            WHERE symbol = :s AND trade_date <= :d
            ORDER BY trade_date DESC LIMIT 1
        """), {"s": symbol, "d": _current_as_of}).fetchone()
    data = dict(row._mapping) if row else {}
    return {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in data.items()}


_ag.fetch_market_data = _patched_fetch_market_data
if HAS_V2:
    _ag2.fetch_market_data = _patched_fetch_market_data


_original_fetch_news = _ag.fetch_news


def _patched_fetch_news(symbol: str, limit: int = 8):
    if _current_as_of is None:
        return _original_fetch_news(symbol, limit)
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, headline, sentiment, source, url
            FROM news_articles
            WHERE headline ILIKE :pat
              AND (published_date IS NULL OR published_date <= :d)
            ORDER BY id DESC LIMIT :n
        """), {"pat": f"%{symbol}%", "d": _current_as_of, "n": limit}).fetchall()
        return [dict(r._mapping) for r in rows]


_ag.fetch_news = _patched_fetch_news
if HAS_V2:
    _ag2.fetch_news = _patched_fetch_news


_original_fetch_rag = _ag.fetch_rag_matches


def _patched_fetch_rag_matches(query: str, symbol: str, top_k: int = 3):
    if _current_as_of is None:
        return _original_fetch_rag(query, symbol, top_k)
    try:
        sim_year = int(_current_as_of[:4])
        res = _ag.pinecone_index.query(
            vector=_ag.embedding_model.encode(query).tolist(),
            top_k=top_k,
            include_metadata=True,
            filter={"symbol": {"$eq": symbol}, "year": {"$lte": sim_year}},
        )
        return [
            {"text": m["metadata"]["text"],
             "source_url": m["metadata"].get("source_url", ""),
             "quarter": m["metadata"].get("quarter", ""),
             "year": m["metadata"].get("year"),
             "symbol": m["metadata"].get("symbol", ""),
             "page_number": m["metadata"].get("page_number", "?")}
            for m in res.get("matches", [])
            if "text" in m.get("metadata", {})
        ]
    except Exception:
        return []


_ag.fetch_rag_matches = _patched_fetch_rag_matches
if HAS_V2:
    _ag2.fetch_rag_matches = _patched_fetch_rag_matches


# ═══════════════════════════════════════════════════════════════════
# DATE HELPERS
# ═══════════════════════════════════════════════════════════════════
def get_trading_dates(ticker: str, start: str, end: str) -> List[str]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT DISTINCT trade_date FROM market_prices_eod
            WHERE symbol = :s AND trade_date BETWEEN :start AND :end
            ORDER BY trade_date
        """), {"s": ticker, "start": start, "end": end}).fetchall()
    return [str(r._mapping["trade_date"]) for r in rows]


def nearest_trading_day(target: str, trading_dates: List[str]) -> Optional[str]:
    if not trading_dates:
        return None
    target_dt = datetime.strptime(target, "%Y-%m-%d").date()
    best = None
    best_diff = None
    for d in trading_dates:
        d_dt = datetime.strptime(d, "%Y-%m-%d").date()
        diff = abs((d_dt - target_dt).days)
        if best_diff is None or diff < best_diff:
            best, best_diff = d, diff
    return best


def pick_evenly_spaced_dates(start: str, end: str, n: int, trading_dates: List[str]) -> List[str]:
    if not trading_dates:
        return []
    start_dt = datetime.strptime(start, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end, "%Y-%m-%d").date()
    total_days = (end_dt - start_dt).days
    if total_days <= 0:
        return [trading_dates[0]] if trading_dates else []
    step = total_days / n
    picks: List[str] = []
    for i in range(n):
        offset = int(step * (i + 0.5))
        target_dt = start_dt + timedelta(days=offset)
        target_str = target_dt.strftime("%Y-%m-%d")
        snapped = nearest_trading_day(target_str, trading_dates)
        if snapped and snapped not in picks:
            picks.append(snapped)
    # Dedup + fill if collisions
    if len(picks) < n:
        for d in trading_dates:
            if d not in picks and start <= d <= end:
                picks.append(d)
                if len(picks) >= n:
                    break
    return sorted(picks)[:n]


# ═══════════════════════════════════════════════════════════════════
# SIGNAL EXTRACTION
# ═══════════════════════════════════════════════════════════════════
def extract_signal(report: str) -> Tuple[str, float, Optional[int], bool]:
    """Returns (verdict, confidence, holding_days, has_verdict)."""
    if not report:
        return "MISSING", 0.0, None, False

    verdict = "MISSING"
    conf = 0.0
    holding = None

    patterns = [
        r"VERDICT[#\*_`\s]*[:\-][\s\*_`]*(BUY|SELL|HOLD)\b",
        r"VERDICT\s+is\s+[\"\']?(BUY|SELL|HOLD)\b",
        r"VERDICT[\"\']?\s*[:=]\s*[\"\']?(BUY|SELL|HOLD)\b",
    ]
    for pat in patterns:
        m = re.search(pat, report, re.IGNORECASE)
        if m:
            verdict = m.group(1).upper()
            break

    cm = re.search(
        r"CONFIDENCE[\s\*_`]*[:\-][\s\*_`]*(0?\.\d+|[01](?:\.\d+)?)",
        report, re.IGNORECASE
    )
    if cm:
        try:
            conf = min(1.0, max(0.0, float(cm.group(1))))
        except ValueError:
            pass

    hm = re.search(r"HOLDING[_\s]*PERIOD[\s\*_`]*[:\-][\s\*_`]*(\d{1,3})",
                   report, re.IGNORECASE)
    if hm:
        try:
            holding = int(hm.group(1))
        except ValueError:
            pass

    has_verdict = verdict != "MISSING"
    return verdict, conf, holding, has_verdict


# ═══════════════════════════════════════════════════════════════════
# GRAPH RUNNER
# ═══════════════════════════════════════════════════════════════════
def run_decision(graph, ticker: str, as_of: str, version: str) -> Dict[str, Any]:
    """Run the graph ONCE for a (ticker, date). Returns decision dict."""
    global _current_as_of
    _current_as_of = as_of
    try:
        state = {
            "query": f"Analyze {ticker} as of {as_of}.",
            "symbol": ticker,
            "numerical_data": {},
            "retrieved_docs": [],
            "sentiment_summary": "",
            "risk_alerts": [],
            "final_report": "",
        }
        final = None
        for event in graph.stream(state):
            if isinstance(event, dict) and "__end__" in event:
                final = event["__end__"]

        if not final:
            return {"verdict": "ERROR", "confidence": 0.0, "holding_days": None,
                    "report": "no final state", "has_verdict": False}

        report = final.get("final_report", "") or ""
        v, c, h, has_v = extract_signal(report)
        return {"verdict": v, "confidence": c, "holding_days": h,
                "report": report, "has_verdict": has_v}
    except SessionLimitReached:
        raise
    except Exception as e:
        logger.exception(f"[{version}] Graph failed for {ticker} @ {as_of}: {e}")
        return {"verdict": "ERROR", "confidence": 0.0, "holding_days": None,
                "report": f"error: {e}", "has_verdict": False}
    finally:
        _current_as_of = None


def run_decision_with_retry(graph, ticker: str, as_of: str, version: str,
                            max_attempts: int = 2) -> Dict[str, Any]:
    last = None
    for attempt in range(1, max_attempts + 1):
        last = run_decision(graph, ticker, as_of, version)
        if last.get("verdict") not in ("ERROR", "MISSING"):
            return last
        if attempt < max_attempts:
            print(f"    ↩ retry {attempt}/{max_attempts-1} for {ticker}@{as_of} "
                  f"[{version}]", flush=True)
            time.sleep(2)
    return last


# ═══════════════════════════════════════════════════════════════════
# MAIN BACKTEST — decide-then-print, no simulation
# ═══════════════════════════════════════════════════════════════════
def run_backtest(
    stocks: List[str],
    dates: List[str],
    progress_cb=None,
) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
        {
          "OGDC": {
            "dates": [...],
            "v1": [{date, verdict, confidence, holding_days, report}, ...],
            "v2": [...],
          },
          ...
        }
    """
    if not HAS_V2:
        raise RuntimeError("agents_graph_v2.py is not importable.")

    graph_v1 = build_v1()
    graph_v2 = build_v2()

    results: Dict[str, Dict[str, Any]] = {}

    for ticker in stocks:
        print(f"\n  ▶ Running {ticker} on {len(dates)} dates…", flush=True)
        entry = {"dates": list(dates), "v1": [], "v2": []}

        for i, date in enumerate(dates):
            if progress_cb:
                progress_cb(ticker, i + 1, len(dates), date)

            d1 = run_decision_with_retry(graph_v1, ticker, date, "v1")
            entry["v1"].append({"date": date, **d1})

            d2 = run_decision_with_retry(graph_v2, ticker, date, "v2")
            entry["v2"].append({"date": date, **d2})

        results[ticker] = entry

    return results


# ═══════════════════════════════════════════════════════════════════
# TERMINAL PRINTERS
# ═══════════════════════════════════════════════════════════════════
def _fmt_verdict(v: str) -> str:
    return f"{v:<6}"


def _fmt_conf(c: float) -> str:
    return f"{c:.2f}"


def _fmt_hold(h: Optional[int]) -> str:
    return f"{h}d" if h is not None else "—"


def _agree(v1: str, v2: str) -> str:
    if v1 == "MISSING" or v2 == "MISSING":
        return "?"
    if v1 == "ERROR" or v2 == "ERROR":
        return "?"
    return "✓" if v1 == v2 else "✗"


def print_per_stock_block(ticker: str, entry: Dict[str, Any]) -> None:
    print("\n" + "═" * 78, flush=True)
    print(f"{ticker} — {len(entry['dates'])} decisions across Jan–Oct 2026", flush=True)
    print("═" * 78, flush=True)
    print(f"  {'Date':<12} {'V1 verdict':<10} {'conf':<6} {'hold':<5} "
          f"{'V2 verdict':<10} {'conf':<6} {'hold':<5} {'Agree':<5}", flush=True)
    print("  " + "─" * 74, flush=True)

    v1_tally = {"BUY": 0, "HOLD": 0, "SELL": 0, "MISSING": 0, "ERROR": 0}
    v2_tally = {"BUY": 0, "HOLD": 0, "SELL": 0, "MISSING": 0, "ERROR": 0}
    v1_conf_sum = v2_conf_sum = 0.0
    agree_count = 0
    total_valid = 0

    for i, date in enumerate(entry["dates"]):
        a = entry["v1"][i]
        b = entry["v2"][i]
        v1_tally[a["verdict"]] = v1_tally.get(a["verdict"], 0) + 1
        v2_tally[b["verdict"]] = v2_tally.get(b["verdict"], 0) + 1
        if a["verdict"] in ("BUY", "HOLD", "SELL"):
            v1_conf_sum += a["confidence"]
        if b["verdict"] in ("BUY", "HOLD", "SELL"):
            v2_conf_sum += b["confidence"]
        agree = _agree(a["verdict"], b["verdict"])
        if agree in ("✓", "✗"):
            total_valid += 1
            if agree == "✓":
                agree_count += 1

        print(f"  {date:<12} {_fmt_verdict(a['verdict']):<10} "
              f"{_fmt_conf(a['confidence']):<6} {_fmt_hold(a['holding_days']):<5} "
              f"{_fmt_verdict(b['verdict']):<10} "
              f"{_fmt_conf(b['confidence']):<6} {_fmt_hold(b['holding_days']):<5} "
              f"{agree:<5}", flush=True)

    v1_valid = v1_tally["BUY"] + v1_tally["HOLD"] + v1_tally["SELL"]
    v2_valid = v2_tally["BUY"] + v2_tally["HOLD"] + v2_tally["SELL"]
    v1_avg = v1_conf_sum / v1_valid if v1_valid else 0
    v2_avg = v2_conf_sum / v2_valid if v2_valid else 0
    agree_pct = (agree_count / total_valid * 100) if total_valid else 0

    print(f"\n  V1 tally: {v1_tally['BUY']} BUY / {v1_tally['HOLD']} HOLD / "
          f"{v1_tally['SELL']} SELL  ·  avg conf {v1_avg:.2f}", flush=True)
    print(f"  V2 tally: {v2_tally['BUY']} BUY / {v2_tally['HOLD']} HOLD / "
          f"{v2_tally['SELL']} SELL  ·  avg conf {v2_avg:.2f}", flush=True)
    if v1_tally["MISSING"] or v1_tally["ERROR"]:
        print(f"  V1 issues: {v1_tally['MISSING']} missing verdict, "
              f"{v1_tally['ERROR']} errors", flush=True)
    if v2_tally["MISSING"] or v2_tally["ERROR"]:
        print(f"  V2 issues: {v2_tally['MISSING']} missing verdict, "
              f"{v2_tally['ERROR']} errors", flush=True)
    print(f"  Agreement: {agree_count}/{total_valid} ({agree_pct:.0f}%)", flush=True)


def print_cross_section(all_results: Dict[str, Dict[str, Any]]) -> None:
    if not all_results:
        return
    any_ticker = next(iter(all_results.values()))
    dates = any_ticker["dates"]

    print("\n\n" + "█" * 78, flush=True)
    print("█ CROSS-SECTION SUMMARY — same date, all stocks, V1 vs V2", flush=True)
    print("█" * 78, flush=True)

    for date in dates:
        print(f"\n  {date}", flush=True)
        print("  " + "─" * 74, flush=True)
        print(f"  {'Stock':<8} {'V1 verdict':<10} {'conf':<6} {'hold':<5} "
              f"{'V2 verdict':<10} {'conf':<6} {'hold':<5} {'Agree':<5}", flush=True)
        for ticker, entry in all_results.items():
            try:
                i = entry["dates"].index(date)
            except ValueError:
                continue
            a = entry["v1"][i]
            b = entry["v2"][i]
            print(f"  {ticker:<8} {_fmt_verdict(a['verdict']):<10} "
                  f"{_fmt_conf(a['confidence']):<6} {_fmt_hold(a['holding_days']):<5} "
                  f"{_fmt_verdict(b['verdict']):<10} "
                  f"{_fmt_conf(b['confidence']):<6} {_fmt_hold(b['holding_days']):<5} "
                  f"{_agree(a['verdict'], b['verdict']):<5}", flush=True)


def print_final_agreement(all_results: Dict[str, Dict[str, Any]]) -> None:
    total = agreed = 0
    for entry in all_results.values():
        for i in range(len(entry["dates"])):
            v1 = entry["v1"][i]["verdict"]
            v2 = entry["v2"][i]["verdict"]
            if v1 in ("BUY", "HOLD", "SELL") and v2 in ("BUY", "HOLD", "SELL"):
                total += 1
                if v1 == v2:
                    agreed += 1

    print("\n" + "█" * 78, flush=True)
    print("█ OVERALL AGREEMENT", flush=True)
    print("█" * 78, flush=True)
    if total:
        pct = agreed / total * 100
        print(f"\n  V1 and V2 agreed on {agreed}/{total} decisions ({pct:.0f}%).", flush=True)
        if pct >= 80:
            print("  ⚠️  High agreement — the two versions behave similarly. "
                  "Comparison may be statistically weak.", flush=True)
        elif pct <= 20:
            print("  ✅ Low agreement — V1 and V2 genuinely diverge. "
                  "Comparison is meaningful.", flush=True)
        else:
            print("  ℹ️  Moderate agreement — real but small difference.", flush=True)
    else:
        print("\n  No valid decisions to compare.", flush=True)
    print("█" * 78 + "\n", flush=True)


# ═══════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Decision viewer: what did V1 and V2 say on each date?"
    )
    parser.add_argument("--stocks", default="OGDC,PSO,LUCK",
                        help="Comma-separated tickers (default: OGDC,PSO,LUCK)")
    parser.add_argument("--start", default="2026-01-01")
    parser.add_argument("--end", default="2026-10-31")
    parser.add_argument("--dates-n", type=int, default=7,
                        help="How many evenly-spaced dates (default: 7)")
    parser.add_argument("--dates", default=None,
                        help="Explicit CSV of dates; overrides --dates-n")
    parser.add_argument("--random", action="store_true",
                        help="Random dates instead of evenly-spaced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rpm", type=int, default=8)
    parser.add_argument("--call-budget", type=int, default=400)
    parser.add_argument("--output", default="backtest_results.json")
    args = parser.parse_args()

    if not HAS_V2:
        print("❌ agents_graph_v2.py not importable. Cannot run V1 vs V2.", flush=True)
        raise SystemExit(1)

    # Install throttle
    limiter = RateLimiter(rpm=args.rpm, daily=args.call_budget)
    print(f"\n⏱️  LLM throttle: {args.rpm}/min, {args.call_budget}/session", flush=True)
    n_patched = install_rate_limiter(limiter)
    if n_patched == 0:
        print("❌ No LLM entry point throttled — aborting.", flush=True)
        raise SystemExit(1)

    stocks = [s.strip().upper() for s in args.stocks.split(",") if s.strip()]
    if not stocks:
        print("❌ No stocks provided.", flush=True)
        raise SystemExit(1)

    # Determine dates
    all_trading_dates: List[str] = []
    for s in stocks:
        all_trading_dates.extend(get_trading_dates(s, args.start, args.end))
    all_trading_dates = sorted(set(all_trading_dates))

    if not all_trading_dates:
        print(f"❌ No trading data for {stocks} between {args.start} and {args.end}.", flush=True)
        raise SystemExit(1)

    if args.dates:
        requested = [d.strip() for d in args.dates.split(",") if d.strip()]
        dates = [nearest_trading_day(d, all_trading_dates) for d in requested]
        dates = [d for d in dates if d]
        # Dedup while preserving order
        seen, dates = set(), []
        for d in dates:
            if d not in seen:
                seen.add(d)
                dates.append(d)
    elif args.random:
        import random as _r
        rng = _r.Random(args.seed)
        n = min(args.dates_n, len(all_trading_dates))
        dates = sorted(rng.sample(all_trading_dates, n))
    else:
        dates = pick_evenly_spaced_dates(
            args.start, args.end, args.dates_n, all_trading_dates,
        )

    if not dates:
        print("❌ Could not determine any valid dates.", flush=True)
        raise SystemExit(1)

    print("\n" + "█" * 78, flush=True)
    print("█ DECISION VIEWER — V1 vs V2, no simulation", flush=True)
    print(f"█ Stocks:       {', '.join(stocks)}", flush=True)
    print(f"█ Range:        {args.start} → {args.end}", flush=True)
    print(f"█ Dates/stock:  {len(dates)}", flush=True)
    print(f"█ Dates:        {', '.join(dates)}", flush=True)
    print(f"█ Rate limit:   {args.rpm}/min, {args.call_budget} total", flush=True)
    print(f"█ Started at:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    est_calls = len(stocks) * len(dates) * 2
    print(f"█ Expected calls: ~{est_calls}  ·  ~{est_calls / max(1, args.rpm):.0f} min at {args.rpm}/min", flush=True)
    print("█" * 78 + "\n", flush=True)

    t_start = time.time()

    def progress(ticker, step, total, date):
        elapsed = (time.time() - t_start) / 60
        print(f"    [{ticker}] {step}/{total} @ {date}  "
              f"(elapsed {elapsed:.1f} min, calls {limiter.calls_used}/{args.call_budget})",
              flush=True)

    try:
        all_results = run_backtest(stocks, dates, progress_cb=progress)
    except SessionLimitReached as e:
        print(f"\n🛑 Session limit reached: {e}", flush=True)
        raise SystemExit(1)

    for ticker in stocks:
        if ticker in all_results:
            print_per_stock_block(ticker, all_results[ticker])

    print_cross_section(all_results)
    print_final_agreement(all_results)

    # Save slim JSON
    payload = {
        "config": {
            "stocks": stocks,
            "start": args.start,
            "end": args.end,
            "dates": dates,
            "dates_n": len(dates),
            "rpm": args.rpm,
            "call_budget": args.call_budget,
        },
        "llm_usage": limiter.summary(),
        "elapsed_seconds": round(time.time() - t_start, 1),
        "completed_at": datetime.now().isoformat(),
        "results": all_results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    print(f"📁 Results saved to: {os.path.abspath(args.output)}", flush=True)
    print(f"⏱️  Total elapsed: {(time.time() - t_start)/60:.1f} minutes", flush=True)
    print(f"📊 LLM usage: {limiter.summary()}\n", flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    main()