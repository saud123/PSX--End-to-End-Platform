# app.py
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi import FastAPI, HTTPException, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import text
from typing import List
from chat_agent import chat_stream
import os
import shutil
import json
import tempfile
import time
import logging
import pandas as pd
import numpy as np
from compare import run_comparative_analysis

import re as _signal_re


# ── Silence psxdata noise via a logging filter (not stdout hijack) ──
class _PsxDataNoiseFilter(logging.Filter):
    """Drops noisy print() lines from psxdata that leak through logging."""
    _DROP = (
        "OHLC constraint violated",
        "Dates are not in chronological order",
        "No historical data returned for",
    )

    def filter(self, record):
        msg = record.getMessage()
        return not any(tag in msg for tag in self._DROP)


logging.getLogger("psxdata").addFilter(_PsxDataNoiseFilter())


def _extract_signal_fields(report_text: str) -> dict:
    if not report_text:
        return {"verdict": "HOLD", "confidence": 0.0, "holding_days": None}

    verdict = "HOLD"
    vm = _signal_re.search(r"VERDICT\s*[:\-]\s*(BUY|SELL|HOLD)\b", report_text, _signal_re.IGNORECASE)
    if vm:
        verdict = vm.group(1).upper()

    confidence = 0.0
    cm = _signal_re.search(r"CONFIDENCE\s*[:\-]\s*(0?\.\d+|[01](?:\.\d+)?)", report_text, _signal_re.IGNORECASE)
    if cm:
        try:
            confidence = min(1.0, max(0.0, float(cm.group(1))))
        except ValueError:
            pass

    holding_days = None
    hm = _signal_re.search(r"HOLDING[_\s]*PERIOD\s*[:\-]\s*(\d{1,3})", report_text, _signal_re.IGNORECASE)
    if hm:
        try:
            holding_days = int(hm.group(1))
        except ValueError:
            pass

    return {"verdict": verdict, "confidence": confidence, "holding_days": holding_days}


AGENT_VERSION = os.getenv("PSX_AGENT_VERSION", "v1").lower()

if AGENT_VERSION == "v2":
    from agents_graph_v2 import build_psx_graph
else:
    from agents_graph import build_psx_graph

# ── Unified DB engine + schema bootstrap ──
from db import engine, init_db
init_db()

from ingestion_report import (
    find_quarterly_reports,
    auto_ingest_psx_report,
    ingest_quarterly_pdf,
)


app = FastAPI(
    title="PSX Multi-Agent Financial Intelligence API",
    description="Autonomous multi-agent pipeline combining Postgres, Pinecone, and LangGraph.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

print("🚀 Initializing LangGraph workflow for FastAPI...")
app_graph = build_psx_graph()


class AnalysisRequest(BaseModel):
    query: str
    symbol: str = ""
    days: int = 180

class AutoIngestRequest(BaseModel):
    symbol: str

class BacktestRequest(BaseModel):
    tickers: List[str]
    start_date: str
    end_date: str
    initial_capital: float = 1_000_000
    rebalance: str = "weekly"

class ChatRequest(BaseModel):
    message: str
    thread_id: str

class CompareRequest(BaseModel):
    symbols: List[str]


# ============================================================
# ML price forecaster — LightGBM on engineered features
# ============================================================
_FEATURE_COLS = [
    "ret_1", "ret_2", "ret_3", "ret_5", "ret_10",
    "vol_5", "vol_10", "vol_20",
    "sma_5_ratio", "sma_20_ratio",
    "rsi_norm", "macd_norm", "bb_pos", "vol_ratio", "dow",
]


def _build_ml_features(df: pd.DataFrame) -> pd.DataFrame:
    """Self-contained tabular features for the LightGBM predictor.
    Every feature at row t uses only data <= t (no lookahead).
    Works even on synthetic appended rows that lack indicator columns.
    """
    out = df.copy()

    if not {"rsi", "macd_hist", "bb_upper", "bb_lower"}.issubset(out.columns):
        delta = out["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        out["rsi"] = 100 - 100 / (1 + gain / loss)
        e12 = out["close"].ewm(span=12, adjust=False).mean()
        e26 = out["close"].ewm(span=26, adjust=False).mean()
        macd = e12 - e26
        out["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
        mid = out["close"].rolling(20).mean()
        sd = out["close"].rolling(20).std()
        out["bb_upper"] = mid + 2 * sd
        out["bb_lower"] = mid - 2 * sd

    close = out["close"]
    log_ret = np.log(close / close.shift(1))
    out["ret_1"]  = log_ret
    out["ret_2"]  = np.log(close / close.shift(2))
    out["ret_3"]  = np.log(close / close.shift(3))
    out["ret_5"]  = np.log(close / close.shift(5))
    out["ret_10"] = np.log(close / close.shift(10))
    out["vol_5"]  = log_ret.rolling(5).std()
    out["vol_10"] = log_ret.rolling(10).std()
    out["vol_20"] = log_ret.rolling(20).std()
    out["sma_5_ratio"]  = close.rolling(5).mean()  / close - 1
    out["sma_20_ratio"] = close.rolling(20).mean() / close - 1
    out["rsi_norm"]  = out["rsi"] / 100.0
    out["macd_norm"] = out["macd_hist"] / close
    rng = (out["bb_upper"] - out["bb_lower"]).replace(0, np.nan)
    out["bb_pos"] = (close - out["bb_lower"]) / rng

    if "volume" in out.columns:
        v = pd.to_numeric(out["volume"], errors="coerce")
        out["vol_ratio"] = v / v.rolling(20).mean()
    else:
        out["vol_ratio"] = 1.0

    out["dow"] = pd.to_datetime(out["trade_date"]).dt.dayofweek
    return out.replace([np.inf, -np.inf], np.nan)


def _ml_forecast(df: pd.DataFrame, horizon: int = 7) -> list:
    """7-day recursive forecast of log-returns via LightGBM (XGBoost fallback)."""
    last_close = float(df["close"].iloc[-1])
    last_date  = pd.to_datetime(df["trade_date"].iloc[-1])
    future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=horizon)

    def _rw():
        return [{
            "trade_date": d.strftime("%Y-%m-%d"),
            "predicted_close": round(last_close, 2),
            "lower_close": round(last_close, 2),
            "upper_close": round(last_close, 2),
        } for d in future_dates]

    feat = _build_ml_features(df)
    feat["target"] = np.log(feat["close"].shift(-1) / feat["close"])
    train = feat.dropna(subset=_FEATURE_COLS + ["target"]).reset_index(drop=True)
    if len(train) < 60:
        print(f"ℹ️ Only {len(train)} usable rows; using random walk.")
        return _rw()

    X = train[_FEATURE_COLS].values
    y = train["target"].values

    model = None
    try:
        from lightgbm import LGBMRegressor
        model = LGBMRegressor(
            n_estimators=400, learning_rate=0.03, max_depth=4,
            num_leaves=15, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.5,
            random_state=42, verbosity=-1,
        )
        model.fit(X, y)
    except Exception as e:
        print(f"⚠️ LightGBM unavailable ({e}); trying XGBoost.")
        try:
            from xgboost import XGBRegressor
            model = XGBRegressor(
                n_estimators=400, learning_rate=0.03, max_depth=4,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbosity=0,
            )
            model.fit(X, y)
        except Exception as e2:
            print(f"⚠️ XGBoost also failed ({e2}); using random walk.")
            return _rw()

    resid = y - model.predict(X)
    resid_std = max(float(np.std(resid)) if len(resid) else 0.02, 1e-4)

    RETURN_CLIP = 0.10
    hist = df.copy().reset_index(drop=True)
    preds = []

    for step, next_date in enumerate(future_dates, start=1):
        row = _build_ml_features(hist).iloc[[-1]][_FEATURE_COLS].fillna(0.0)
        pred_ret = float(np.clip(model.predict(row.values)[0],
                                 -RETURN_CLIP, RETURN_CLIP))

        prev_close = float(hist["close"].iloc[-1])
        next_close = max(0.1, prev_close * float(np.exp(pred_ret)))
        band       = 1.96 * resid_std * np.sqrt(step)

        preds.append({
            "trade_date":     next_date.strftime("%Y-%m-%d"),
            "predicted_close": round(next_close, 2),
            "lower_close":     round(max(0.1, next_close * np.exp(-band)), 2),
            "upper_close":     round(next_close * np.exp(band), 2),
        })

        vol = (float(hist["volume"].iloc[-1])
               if "volume" in hist.columns and pd.notna(hist["volume"].iloc[-1])
               else 0.0)
        hist = pd.concat([hist, pd.DataFrame([{
            "trade_date": next_date, "open": next_close, "high": next_close,
            "low": next_close, "close": next_close, "volume": vol,
        }])], ignore_index=True)

    return preds


def get_enriched_history(symbol: str, limit: int = 180):
    fetch_limit = max(limit + 150, 500)
    with engine.connect() as conn:
        res = conn.execute(
            text(
                "SELECT trade_date, open, high, low, close, volume "
                "FROM market_prices_eod WHERE symbol = :s "
                "ORDER BY trade_date DESC LIMIT :lim"
            ),
            {"s": symbol.upper(), "lim": fetch_limit},
        )
        rows = [dict(r._mapping) for r in res]

    if not rows:
        return {"history": [], "prediction": []}

    df = pd.DataFrame(rows).sort_values("trade_date").reset_index(drop=True)
    df["trade_date"] = pd.to_datetime(df["trade_date"])

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    df["rsi"] = 100 - (100 / (1 + (gain / loss)))

    exp12 = df["close"].ewm(span=12, adjust=False).mean()
    exp26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = exp12 - exp26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    df["bb_mid"] = df["close"].rolling(window=20).mean()
    df["bb_std"] = df["close"].rolling(window=20).std()
    df["bb_upper"] = df["bb_mid"] + (df["bb_std"] * 2)
    df["bb_lower"] = df["bb_mid"] - (df["bb_std"] * 2)

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    prediction_results = []
    try:
        prediction_results = _ml_forecast(df, horizon=7)
    except Exception as e:
        print(f"⚠️ ML forecast failed: {e}")

    df = df.tail(limit)
    df = df.fillna(0)
    df["trade_date"] = df["trade_date"].dt.strftime("%Y-%m-%d")
    return {"history": df.to_dict("records"), "prediction": prediction_results}


# ============================================================
# Routes
# ============================================================
@app.get("/")
def read_root():
    """Serve the SPA shell. Hash router inside takes over from here."""
    if not os.path.exists("static/index.html"):
        return JSONResponse(status_code=404, content={"message": "index.html not found in static/"})
    return FileResponse("static/index.html")


@app.get("/index.html")
def read_index_html():
    """Serve the SPA shell at its explicit filename (used by Home.html links)."""
    if not os.path.exists("static/index.html"):
        return JSONResponse(status_code=404, content={"message": "index.html not found in static/"})
    return FileResponse("static/index.html")


@app.get("/home")
@app.get("/Home.html")
@app.get("/home.html")
def read_home():
    """Serve the marketing landing page."""
    if not os.path.exists("static/Home.html"):
        return JSONResponse(status_code=404, content={"message": "Home.html not found in static/"})
    return FileResponse("static/Home.html")


# SPA assets — index.html references them relatively (`styles.css`, `app.js`),
# so they need to be reachable at root, not only under /static/.
@app.get("/styles.css")
def serve_styles():
    return FileResponse("static/styles.css", media_type="text/css")


@app.get("/app.js")
def serve_app_js():
    return FileResponse("static/app.js", media_type="application/javascript")


# ── Legacy multi-page routes → redirect to SPA hash routes ──
# Keeps old bookmarks working without maintaining separate HTML files.
@app.get("/chat")
def read_chat():
    return RedirectResponse(url="/#/chat", status_code=302)


@app.get("/compare")
def read_compare():
    return RedirectResponse(url="/#/compare", status_code=302)


@app.get("/quant")
def read_quant():
    return RedirectResponse(url="/#/screen", status_code=302)


@app.get("/about")
def read_about():
    return RedirectResponse(url="/", status_code=302)


@app.get("/backtest")
def read_backtest():
    return RedirectResponse(url="/", status_code=302)


@app.get("/health")
def health_check():
    return {"status": "online", "system": "PSX Multi-Agent Pipeline", "agent_version": AGENT_VERSION}


# ── Quant API ──
@app.get("/api/quant/top-picks")
def get_quant_top_picks(
    n: int = 10,
    profile: str = "balanced",
    refresh: bool = False,
):
    from quant_screener import run_quant_screener
    n = max(1, min(int(n), 25))
    profile = profile if profile in ("balanced", "aggressive", "conservative") else "balanced"
    return run_quant_screener(top_n=n, profile=profile, force_refresh=refresh)


@app.get("/api/market/movers")
def get_market_movers():
    try:
        with engine.connect() as conn:
            vol_rows = conn.execute(text("""
                SELECT symbol, volume 
                FROM market_prices_eod 
                WHERE trade_date = (SELECT MAX(trade_date) FROM market_prices_eod)
                ORDER BY volume DESC LIMIT 3
            """)).fetchall()

            vola_rows = conn.execute(text("""
                WITH RankedPrices AS (
                    SELECT symbol, close, 
                           LAG(close) OVER (PARTITION BY symbol ORDER BY trade_date) AS prev_close,
                           trade_date
                    FROM market_prices_eod
                )
                SELECT symbol, close, prev_close, 
                       ROUND(CAST(((close - prev_close) / prev_close) * 100 AS numeric), 2) as change_pct
                FROM RankedPrices
                WHERE trade_date = (SELECT MAX(trade_date) FROM market_prices_eod)
                  AND prev_close IS NOT NULL AND prev_close > 0
                ORDER BY ABS(((close - prev_close) / prev_close) * 100) DESC LIMIT 3
            """)).fetchall()

        
        return {
            "volume": [dict(r._mapping) for r in vol_rows],
            "volatility": [dict(r._mapping) for r in vola_rows]
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})


@app.get("/api/stocks/{symbol}/history")
def get_stock_history(symbol: str, limit: int = 180):
    limit = max(30, min(int(limit), 730))
    return {"symbol": symbol.upper(), **get_enriched_history(symbol, limit)}


# ── Fail-hard admin token (no default) ──
ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    raise RuntimeError(
        "ADMIN_TOKEN environment variable must be set (no default for security). "
        "Export it in your shell or .env before starting the server."
    )


@app.post("/api/admin/upload-report")
async def upload_financial_report(
    symbol: str = Form(...),
    quarter: str = Form(...),
    year: int = Form(...),
    file: UploadFile = File(...),
    x_admin_token: str = Header(None)
):
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid Admin Token")

    # Safe temp file — ignores client filename, no path traversal possible
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_path = tmp.name
    try:
        with tmp:
            shutil.copyfileobj(file.file, tmp)
        ingest_quarterly_pdf(temp_path, symbol=symbol, quarter=quarter, year=year)
        return JSONResponse(status_code=200, content={
            "status": "success",
            "message": f"Successfully ingested {quarter} {year} report for {symbol}."
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "status": "error",
            "message": f"Ingestion failed: {str(e)}"
        })
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


@app.get("/api/reports/suggest/{symbol}")
def suggest_reports(symbol: str, limit: int = 8):
    try:
        return {"symbol": symbol.upper(), "suggestions": find_quarterly_reports(symbol, limit=limit)}
    except Exception as e:
        return {"symbol": symbol.upper(), "suggestions": [], "error": str(e)}


@app.post("/api/reports/auto-ingest")
def auto_ingest_report(req: AutoIngestRequest):
    try:
        result = auto_ingest_psx_report(req.symbol)
        if result.get("pdf_ingested"):
            msg = (f"Ingested {result.get('quarter')} {result.get('year')} PDF "
                   f"({result.get('pages', 0)} pages, {result.get('chunks', 0)} chunks) "
                   f"+ {result.get('metrics_ingested', 0)} metrics docs for {result['symbol']}.")
        elif int(result.get("metrics_ingested", 0) or 0) > 0:
            msg = (f"No fresh quarterly PDF (previous-year only). "
                   f"Ingested {result['metrics_ingested']} metrics document(s) for {result['symbol']}.")
        else:
            msg = f"Nothing ingested for {result['symbol']}: {result.get('reason', 'unknown reason')}"
        return JSONResponse(status_code=200, content={"status": "success", "message": msg, "details": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": f"Auto-ingest failed: {str(e)}"})


@app.post("/api/chat/stream")
def api_chat_stream(request: ChatRequest):
    def gen():
        try:
            for chunk in chat_stream(request.thread_id, request.message):
                yield chunk
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache",
                 "Connection": "keep-alive",
                 "X-Accel-Buffering": "no"},
    )


@app.post("/api/compare")
def api_compare(request: CompareRequest):
    try:
        symbols = [s.upper() for s in request.symbols]
        if not (2 <= len(symbols) <= 5):
            raise HTTPException(status_code=400, detail="Must provide between 2 and 5 symbols.")
        return run_comparative_analysis(symbols)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e), "traceback": traceback.format_exc()})


@app.post("/api/analyze")
def run_market_analysis(request: AnalysisRequest):
    try:
        initial_state = {"query": request.query, "symbol": request.symbol, "numerical_data": {}, "retrieved_docs": [], "sentiment_summary": "", "risk_alerts": [], "final_report": ""}
        result_state = app_graph.invoke(initial_state)
        final_report_text = result_state.get("final_report") or ""
        return {
            "success": True,
            "symbol": result_state.get("symbol"),
            "numerical_data": result_state.get("numerical_data"),
            "sentiment_summary": result_state.get("sentiment_summary"),
            "risk_alerts": result_state.get("risk_alerts"),
            "final_report": final_report_text,
            "data_sources": result_state.get("data_sources", {}),
            "signal_summary": _extract_signal_fields(final_report_text),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/analyze/stream")
def run_market_analysis_stream(request: AnalysisRequest):
    def event_generator():
        initial_state = {"query": request.query, "symbol": request.symbol, "numerical_data": {}, "retrieved_docs": [], "sentiment_summary": "", "risk_alerts": [], "final_report": ""}
        accumulated_state = dict(initial_state)
        data_sources: dict = {}
        completed = False
        last_ping = time.time()

        try:
            for event in app_graph.stream(initial_state, stream_mode="updates"):
                # Heartbeat every 5s so proxies/clients don't time out during slow tools
                if time.time() - last_ping > 5:
                    yield "event: ping\ndata: {}\n\n"
                    last_ping = time.time()

                if not isinstance(event, dict):
                    continue
                if "__end__" in event:
                    end_data = event["__end__"]
                    if isinstance(end_data, dict):
                        accumulated_state.update(end_data)
                        data_sources = end_data.get("data_sources", {})
                    continue
                for node_name, node_output in event.items():
                    if isinstance(node_output, dict):
                        accumulated_state.update(node_output)
                    elif isinstance(node_output, list):
                        for sub in node_output:
                            if isinstance(sub, dict):
                                accumulated_state.update(sub)
                    yield "data: " + json.dumps({"type": "node_update", "node": node_name, "status": "completed"}) + "\n\n"

            symbol = accumulated_state.get("symbol")
            chart_data = {"history": [], "prediction": []}
            if symbol:
                try:
                    days = int(request.days)
                    days = days if 30 <= days <= 730 else 180
                    chart_data = get_enriched_history(symbol, limit=days)
                except Exception as hist_err:
                    print(f"⚠️ History fetch failed: {hist_err}")

            final_report_text = accumulated_state.get("final_report") or ""
            final_payload = {
                "type": "complete",
                "symbol": accumulated_state.get("symbol"),
                "numerical_data": accumulated_state.get("numerical_data") or {},
                "sentiment_summary": accumulated_state.get("sentiment_summary"),
                "risk_alerts": accumulated_state.get("risk_alerts") or [],
                "final_report": final_report_text,
                "retrieved_docs": accumulated_state.get("retrieved_docs") or [],
                "price_history": chart_data["history"],
                "prediction": chart_data["prediction"],
                "data_sources": data_sources,
                "agent_version": accumulated_state.get("version", "v1"),
                "signal_summary": _extract_signal_fields(final_report_text),
            }
            yield "data: " + json.dumps(final_payload, default=str) + "\n\n"
            yield "event: complete\ndata: {}\n\n"
            completed = True
        except GeneratorExit:
            print("ℹ️ SSE client disconnected — stopping generator", flush=True)
            return
        except Exception as e:
            logging.exception("SSE stream failed")
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"
            return
        finally:
            if not completed:
                try:
                    yield "event: error\ndata: " + json.dumps({"message": "stream ended without result"}) + "\n\n"
                except Exception:
                    pass

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})


@app.post("/api/backtest/stream")
def backtest_stream(request: BacktestRequest):
    from backtest import run_backtest_pair
    import queue, threading

    def event_generator():
        q: "queue.Queue" = queue.Queue()
        holder = {"data": None, "error": None}

        def worker():
            try:
                def cb(version, step, total, ticker, date):
                    q.put({"type": "progress", "version": version, "step": step, "total": total, "ticker": ticker, "date": date})
                holder["data"] = run_backtest_pair(
                    tickers=request.tickers,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    initial_capital=request.initial_capital,
                    rebalance=request.rebalance,
                    progress_cb=cb,
                )
            except Exception as e:
                logging.exception("Backtest failed")
                holder["error"] = str(e)
            finally:
                q.put({"type": "done"})

        threading.Thread(target=worker, daemon=True).start()

        try:
            while True:
                try:
                    item = q.get(timeout=30)
                except queue.Empty:
                    yield "event: ping\ndata: {}\n\n"
                    continue
                if item["type"] == "done":
                    if holder["error"]:
                        yield "event: error\ndata: " + json.dumps({"message": holder["error"]}) + "\n\n"
                    else:
                        yield "data: " + json.dumps({"type": "complete", "result": holder["data"]}, default=str) + "\n\n"
                        yield "event: complete\ndata: {}\n\n"
                    return
                yield "data: " + json.dumps(item, default=str) + "\n\n"
        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"message": str(e)}) + "\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})