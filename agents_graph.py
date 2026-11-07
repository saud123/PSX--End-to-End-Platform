"""
PSX multi-agent intelligence brief — Agentic Supervisor (Multi-Model Fallback).
V1: Supervisor + specialist agents, with a single synchronous
    download-and-ingest agent for quarterly PDFs.

Market data: safe-hybrid SQL — LLM only extracts "N days" as an integer,
the actual SQL template is hardcoded and auditable.

Parallel execution: when the supervisor calls multiple specialist agents
in one turn, they run concurrently in a thread pool. Wall-clock time
drops from ~sum(latencies) to ~max(latencies).
"""

import os
import re
import json
import time
import traceback
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from sqlalchemy import text

# ✅ Unified LLM router — raises LLMUnavailable on total failure
from llm import generate_text, LLMUnavailable, MODEL_GEMINI as MODEL

# ✅ Unified DB module — single engine + loader
from db import engine, load_psx_market_prices

load_dotenv()

# Gemini client is only used here for the tool-calling supervisor loop.
# All non-tool LLM text goes through llm.generate_text().
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("psx-financial-intelligence")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

NEWS_SOURCE_HOME = {
    "Business Recorder": "https://www.brecorder.com",
    "Investify News": "https://www.investify.pk",
    "Dawn Business": "https://www.dawn.com/newspaper/business-finance",
}

# ============================================================
# MARKET DATA WINDOW — Option A (Safe Hybrid)
# ============================================================
DEFAULT_MARKET_DAYS = 90
MAX_MARKET_DAYS = 3650  # 10 years cap

# Parallelism cap — keeps Gemini/DB/Pinecone from being hammered
MAX_PARALLEL_TOOLS = 6


class GraphState(TypedDict):
    query: str
    symbol: str
    numerical_data: Dict[str, Any]
    retrieved_docs: List[str]
    sentiment_summary: str
    risk_alerts: List[str]
    final_report: str


_VALID_SYMBOLS = None


def get_valid_symbols():
    global _VALID_SYMBOLS
    if _VALID_SYMBOLS is not None:
        return _VALID_SYMBOLS
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("SELECT DISTINCT symbol FROM market_prices_eod")).fetchall()
            _VALID_SYMBOLS = {str(r[0]).upper() for r in rows}
    except Exception:
        _VALID_SYMBOLS = set()
    return _VALID_SYMBOLS


def extract_symbol_from_query(query: str, default_symbol: str = None) -> str:
    if not query:
        return default_symbol
    words = re.findall(r"\b[A-Za-z]{2,7}\b", query.upper())
    valid = get_valid_symbols()
    for w in words:
        if w in valid:
            return w
    return default_symbol


# ------------------------------------------------------------
# LLM extracts "N days" from the user's query.
# ------------------------------------------------------------
def _parse_days_from_query(query: str, default_days: int = DEFAULT_MARKET_DAYS) -> int:
    if not query or not query.strip():
        return default_days

    prompt = f"""Extract how many days of market data the user wants.

User query: "{query}"

Translate common phrases to days:
- "last week" / "past week" → 7
- "last month" / "past month" → 30
- "last quarter" / "past quarter" / "3 months" → 90
- "6 months" / "half year" → 180
- "last year" / "past year" / "12 months" / "YTD" → 365
- "2 years" → 730
- a specific number like "180 days" → 180
- If nothing about time is mentioned, or you are unsure → return {default_days}

Return ONLY a single integer. No words, no JSON, no markdown, no explanation.

Days:"""

    try:
        raw = generate_text(prompt, "You extract integers from text.")
        m = re.search(r"\b(\d{1,5})\b", raw or "")
        if m:
            n = int(m.group(1))
            return max(1, min(n, MAX_MARKET_DAYS))
    except LLMUnavailable as e:
        print(f"⚠️ Days extraction skipped ({e}); using default {default_days}.")
    except Exception as e:
        print(f"⚠️ Days extraction failed ({e}); using default {default_days}.")
    return default_days


# ------------------------------------------------------------
# Fetch a sliding window of prices using a FIXED SQL template.
# ------------------------------------------------------------
def _fetch_market_window(symbol: str, days: int) -> Dict[str, Any]:
    days = max(1, min(int(days), MAX_MARKET_DAYS))

    needs_scrape = False
    try:
        with engine.connect() as conn:
            probe = conn.execute(text("""
                SELECT trade_date
                FROM market_prices_eod
                WHERE symbol = :s
                ORDER BY trade_date DESC LIMIT 1
            """), {"s": symbol}).fetchone()
        if not probe:
            needs_scrape = True
        else:
            latest_date = datetime.strptime(str(probe._mapping["trade_date"]), "%Y-%m-%d").date()
            if (datetime.today().date() - latest_date).days > 3:
                needs_scrape = True
    except Exception:
        needs_scrape = True

    if needs_scrape:
        try:
            load_psx_market_prices(target_symbol=symbol)
        except Exception as e:
            print(f"⚠️ Scraper fallback failed: {e}")

    # days is integer-validated above — safe to embed in SQL via interval
    sql = text("""
        SELECT trade_date, open, high, low, close, volume
        FROM market_prices_eod
        WHERE symbol = :s
          AND trade_date >= CURRENT_DATE - (:days || ' days')::interval
        ORDER BY trade_date DESC
        LIMIT :lim
    """)

    try:
        with engine.connect() as conn:
            rows = conn.execute(sql, {"s": symbol, "days": days, "lim": days * 2}).fetchall()
        history = [dict(r._mapping) for r in rows]
    except Exception as e:
        print(f"⚠️ Window query failed: {e}")
        history = []

    for row in history:
        if hasattr(row.get("trade_date"), "isoformat"):
            row["trade_date"] = row["trade_date"].isoformat()

    latest_row = history[0] if history else {}

    return {
        "symbol": symbol,
        "days": days,
        "latest_row": latest_row,
        "history": history,
        "count": len(history),
    }


def fetch_market_data(symbol: str) -> Dict[str, Any]:
    with engine.connect() as conn:
        row = conn.execute(text("""
            SELECT trade_date, close, volume
            FROM market_prices_eod
            WHERE symbol = :s
            ORDER BY trade_date DESC LIMIT 1
        """), {"s": symbol}).fetchone()
    data = dict(row._mapping) if row else {}

    needs_scrape = False
    if not data:
        needs_scrape = True
    else:
        latest = datetime.strptime(str(data["trade_date"]), "%Y-%m-%d").date()
        if (datetime.today().date() - latest).days > 3:
            needs_scrape = True

    if needs_scrape:
        try:
            load_psx_market_prices(target_symbol=symbol)
            with engine.connect() as conn:
                row = conn.execute(text("""
                    SELECT trade_date, close, volume
                    FROM market_prices_eod
                    WHERE symbol = :s
                    ORDER BY trade_date DESC LIMIT 1
                """), {"s": symbol}).fetchone()
            data = dict(row._mapping) if row else {}
        except Exception as e:
            print(f"⚠️ Scraper fallback failed: {e}")

    return {k: (str(v) if not isinstance(v, (int, float)) else v) for k, v in data.items()}


def fetch_news(symbol: str, limit: int = 8) -> List[Dict[str, Any]]:
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, headline, sentiment, source, url
            FROM news_articles
            WHERE headline ILIKE :pat
            ORDER BY id DESC LIMIT :n
        """), {"pat": f"%{symbol}%", "n": limit}).fetchall()
    return [dict(r._mapping) for r in rows]


def fetch_rag_docs(query: str, symbol: str, top_k: int = 2) -> List[str]:
    try:
        emb = embedding_model.encode(query).tolist()
        res = pinecone_index.query(
            vector=emb, top_k=top_k, include_metadata=True,
            filter={"symbol": {"$eq": symbol}},
        )
        return [
            m["metadata"]["text"]
            for m in res.get("matches", [])
            if "metadata" in m and "text" in m["metadata"]
        ]
    except Exception as e:
        return [f"Vector retrieval error: {e}"]


def fetch_multi_query_rag(symbol: str) -> List[str]:
    queries = [
        f"{symbol} statement of profit or loss sales revenue gross profit earnings per share",
        f"{symbol} statement of financial position assets liabilities equity",
        f"{symbol} statement of cash flows operating investing financing activities",
    ]
    all_docs, seen_texts = [], set()
    for q in queries:
        docs = fetch_rag_docs(q, symbol, top_k=1)
        for d in docs:
            if d and d not in seen_texts and not d.startswith("Vector retrieval error"):
                seen_texts.add(d)
                all_docs.append(d)
    return all_docs


def fetch_rag_matches(query: str, symbol: str, top_k: int = 3) -> List[Dict[str, Any]]:
    try:
        emb = embedding_model.encode(query).tolist()
        res = pinecone_index.query(
            vector=emb, top_k=top_k, include_metadata=True,
            filter={"symbol": {"$eq": symbol}},
        )
        matches: List[Dict[str, Any]] = []
        for m in res.get("matches", []):
            meta = m.get("metadata", {}) or {}
            if "text" not in meta:
                continue
            matches.append({
                "text": meta.get("text", ""),
                "source_url": meta.get("source_url", ""),
                "quarter": meta.get("quarter", ""),
                "year": meta.get("year"),
                "symbol": meta.get("symbol", ""),
                "page_number": meta.get("page_number", "?"),
            })
        return matches
    except Exception as e:
        print(f"⚠️ Pinecone query failed: {e}")
        return []


def rule_based_risk_checks(symbol: str, market_data: Dict[str, Any]) -> List[str]:
    alerts = []
    if not market_data:
        alerts.append(f"No quantitative price data available for {symbol}.")
    else:
        try:
            if float(market_data.get("close", 0)) <= 0:
                alerts.append(f"Invalid close price detected for {symbol} (≤ 0).")
        except (TypeError, ValueError):
            alerts.append(f"Unparseable close price for {symbol}.")
    return alerts


# ============================================================
# SPECIALIST TOOLS
# ============================================================

def market_data_agent(symbol: str, query: str = "") -> Dict[str, Any]:
    days = _parse_days_from_query(query) if query else DEFAULT_MARKET_DAYS
    window = _fetch_market_window(symbol, days)
    latest = window["latest_row"] or {}
    history = window["history"] or []

    closes = [float(r["close"]) for r in history if r.get("close") is not None]
    highs = [float(r["high"]) for r in history if r.get("high") is not None]
    lows = [float(r["low"]) for r in history if r.get("low") is not None]
    vols = [int(r["volume"]) for r in history if r.get("volume") is not None]

    stats: Dict[str, Any] = {
        "window_days": days,
        "rows": len(history),
        "period_high": round(max(highs), 2) if highs else None,
        "period_low": round(min(lows), 2) if lows else None,
        "avg_close": round(sum(closes) / len(closes), 2) if closes else None,
        "avg_volume": int(sum(vols) / len(vols)) if vols else None,
    }
    if len(closes) >= 2 and closes[0]:
        stats["pct_change_over_window"] = round((closes[-1] - closes[0]) / closes[0] * 100, 2)
    else:
        stats["pct_change_over_window"] = None

    prompt = (
        f"You are a PSX Financial Market Data Analyst.\n"
        f"Ticker: {symbol}\n"
        f"Analysis window: last {days} calendar days\n"
        f"Latest EOD row: {latest}\n"
        f"Window stats: {stats}\n\n"
        "Cite the numbers. "
        "Do not invent values. Do NOT use tables."
    )

    try:
        llm_text = generate_text(prompt, "You are a precise financial analyst.")
    except LLMUnavailable as e:
        print(f"⚠️ market_data_agent LLM skipped: {e}")
        llm_text = (
            f"{symbol} · last {days} days: "
            f"avg_close={stats.get('avg_close')}, "
            f"period_high={stats.get('period_high')}, "
            f"period_low={stats.get('period_low')}, "
            f"pct_change={stats.get('pct_change_over_window')}%."
        )

    return {
        "text": llm_text,
        "sources": {
            "agent": "market_data_agent",
            "table": "market_prices_eod",
            "symbol": symbol,
            "days": days,
            "stats": stats,
            "latest_row": latest,
            "source_links": [
                {"label": "PSX Portal", "url": f"https://dps.psx.com.pk/company/{symbol}"}
            ],
        },
    }


def sentiment_agent(symbol: str) -> Dict[str, Any]:
    articles = fetch_news(symbol)
    if not articles:
        return {
            "text": f"No recent news available for {symbol}.",
            "sources": {"agent": "sentiment_agent", "news_count": 0, "source_links": []},
        }

    headlines_block = "\n".join(
        f"{i+1}. {a['headline']}" for i, a in enumerate(articles) if a.get("headline")
    )
    prompt = f"""You are an expert PSX Financial News Sentiment Analyst.
Analyze the following news headlines specifically for their potential market impact on ticker: {symbol}.

Headlines:
{headlines_block}

INSTRUCTIONS:
1. Classify each headline independently as "Positive", "Negative", or "Neutral".
2. Ensure the "labels" array has a strict 1-to-1 correspondence with the headlines.
. Return ONLY valid raw JSON. Do NOT wrap the JSON in markdown code blocks.

Required JSON format:
{{
  "labels": ["Positive" | "Negative" | "Neutral", ...],
  "summary": "2-sentence overall sentiment summary here."
}}"""

    try:
        llm_text = generate_text(prompt, "You are a financial sentiment analyst.")
    except LLMUnavailable as e:
        print(f"⚠️ sentiment_agent LLM skipped: {e}")
        return {
            "text": f"Sentiment classification unavailable for {symbol} (all LLM providers down).",
            "sources": {
                "agent": "sentiment_agent",
                "table": "news_articles",
                "symbol": symbol,
                "news_count": len(articles),
                "articles": [
                    {"headline": (a.get("headline") or "")[:180],
                     "label": "Neutral",
                     "source": a.get("source") or "",
                     "url": a.get("url", "")}
                    for a in articles
                ],
                "error": str(e)[:200],
                "source_links": [],
            },
        }

    try:
        clean_text = llm_text.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean_text)
        labels = parsed.get("labels", [])
        summary = parsed.get("summary", llm_text)
        if len(labels) != len(articles):
            labels = labels + ["Neutral"] * (len(articles) - len(labels))
            labels = labels[: len(articles)]
    except json.JSONDecodeError:
        labels, summary = ["Neutral"] * len(articles), llm_text

    valid = {"Positive", "Negative", "Neutral"}
    try:
        with engine.begin() as conn:
            for art, label in zip(articles, labels):
                if label in valid:
                    conn.execute(
                        text("UPDATE news_articles SET sentiment = :s WHERE id = :id"),
                        {"s": label, "id": art["id"]},
                    )
    except Exception as e:
        print(f"⚠️ Could not persist sentiment labels: {e}")

    articles_out, seen_sources = [], []
    for i, a in enumerate(articles):
        label = labels[i] if i < len(labels) and labels[i] in valid else "Neutral"
        src = a.get("source") or ""
        articles_out.append({
            "headline": (a.get("headline") or "")[:180],
            "label": label,
            "source": src,
            "url": a.get("url", ""),
        })
        if src and src not in seen_sources:
            seen_sources.append(src)

    source_links = [
        {"label": s, "url": NEWS_SOURCE_HOME[s]} for s in seen_sources if s in NEWS_SOURCE_HOME
    ]

    return {
        "text": summary,
        "sources": {
            "agent": "sentiment_agent",
            "table": "news_articles",
            "symbol": symbol,
            "news_count": len(articles),
            "articles": articles_out,
            "source_links": source_links,
        },
    }


def risk_agent(symbol: str) -> Dict[str, Any]:
    market_data = fetch_market_data(symbol)
    alerts = rule_based_risk_checks(symbol, market_data)
    prompt = (
        f"You are a PSX Risk Analyst.\nTicker: {symbol}\n"
        f"Rule-based alerts: {alerts}\n\n"
        "List 3-6 material risks as short bullet points. Cover regulatory, market, "
        "macroeconomic dimensions, and liquidity dimensions where relevant. "
        "Do NOT invent news events."
    )

    try:
        llm_text = generate_text(prompt, "You are a risk analyst.")
    except LLMUnavailable as e:
        print(f"⚠️ risk_agent LLM skipped: {e}")
        llm_text = "\n".join(f"- {a}" for a in alerts) if alerts else "Risk narrative unavailable (LLM down)."

    return {
        "text": llm_text,
        "sources": {"agent": "risk_agent", "alerts": alerts, "symbol": symbol, "source_links": []},
    }


def macro_agent() -> Dict[str, Any]:
    try:
        with engine.connect() as conn:
            res = conn.execute(text(
                "SELECT page_category, chart_title, jsonb_strip_nulls(raw_data) as data "
                "FROM finance_macro_dashboard ORDER BY id DESC LIMIT 20"
            ))
            rows = [dict(r._mapping) for r in res]
        context_lines, seen = [], set()
        for row in rows:
            key = (row["page_category"], row["chart_title"])
            if key in seen:
                continue
            seen.add(key)
            context_lines.append(
                f"- {row['page_category'].replace('-', ' ').title()} / {row['chart_title']}: {row['data']}"
            )
        return {
            "text": "\n".join(context_lines) if context_lines else "No macro data available.",
            "sources": {
                "agent": "macro_agent",
                "table": "finance_macro_dashboard",
                "charts": len(seen),
                "source_links": [{"label": "Ministry of Finance", "url": "https://economy.finance.gov.pk"}],
            },
        }
    except Exception as e:
        return {
            "text": f"Macro data retrieval error: {e}",
            "sources": {"agent": "macro_agent", "error": str(e), "source_links": []},
        }


def rag_agent(symbol: str, query: str = "") -> Dict[str, Any]:
    if not symbol:
        return {
            "text": "RAG agent requires a ticker symbol.",
            "sources": {"agent": "rag_agent", "error": "missing_symbol", "source_links": []},
            "chunks": [],
        }

    docs = fetch_multi_query_rag(symbol)

    if not docs:
        return {
            "text": f"No relevant PDF documents found for {symbol} in the vector store.",
            "sources": {
                "agent": "rag_agent",
                "symbol": symbol,
                "empty": True,
                "source_links": [
                    {"label": f"PSX Filings for {symbol}", "url": f"https://dps.psx.com.pk/company/{symbol}"}
                ],
            },
            "chunks": [],
        }

    text = "\n\n".join(docs)
    source_links = []
    for m in fetch_rag_matches(f"{symbol} quarterly report", symbol):
        if (m.get("source_url") or "").startswith("http"):
            source_links.append({
                "label": f"PSX Report · {m.get('quarter', '')} {m.get('year', '')} (Page {m.get('page_number', '?')})",
                "url": m["source_url"],
            })
            break
    if not source_links:
        source_links.append({
            "label": f"PSX Filings for {symbol}",
            "url": f"https://dps.psx.com.pk/company/{symbol}",
        })

    return {
        "text": text,
        "sources": {
            "agent": "rag_agent",
            "symbol": symbol,
            "chunks_count": len(docs),
            "source_links": source_links,
        },
        "chunks": docs,
    }


def download_report_agent_tool(symbol: str) -> Dict[str, Any]:
    try:
        from ingestion_report import download_report_agent

        result = download_report_agent(symbol)
        docs = fetch_multi_query_rag(symbol)

        pdf_ok     = bool(result.get("pdf_ingested"))
        metrics_ok = int(result.get("metrics_ingested", 0) or 0) > 0

        if pdf_ok:
            text = (
                f"Downloaded and ingested the {result.get('quarter')} "
                f"{result.get('year')} quarterly report for {symbol}. "
                f"{result.get('pages', 0)} pages, "
                f"{result.get('ocr_pages', 0)} OCR'd, "
                f"{result.get('chunks', 0)} chunks embedded into Pinecone. "
                f"Also refreshed {result.get('metrics_ingested', 0)} metrics docs."
            )
            link = {
                "label": f"Downloaded PDF ({result.get('quarter')} {result.get('year')})",
                "url": result.get("source_url") or f"https://dps.psx.com.pk/company/{symbol}",
            }
        elif metrics_ok:
            text = (
                f"No fresh quarterly PDF found for {symbol} (previous-year only). "
                f"Instead ingested {result['metrics_ingested']} metrics document(s) "
                f"(Financials + Ratios from the PSX company page) into Pinecone."
            )
            link = {
                "label": f"PSX company page ({symbol})",
                "url": f"https://dps.psx.com.pk/company/{symbol}",
            }
        else:
            text = (
                f"No PDF or metrics could be ingested for {symbol}: "
                f"{result.get('reason', 'unknown reason')}"
            )
            link = {
                "label": f"PSX Filings for {symbol}",
                "url": f"https://dps.psx.com.pk/company/{symbol}",
            }

        return {
            "text": text,
            "sources": {
                "agent": "download_report_agent_tool",
                "symbol": symbol,
                "pdf_ingested": pdf_ok,
                "metrics_ingested": result.get("metrics_ingested", 0),
                "quarter": result.get("quarter"),
                "year": result.get("year"),
                "pages": result.get("pages"),
                "ocr_pages": result.get("ocr_pages"),
                "chunks": result.get("chunks"),
                "source_links": [link],
            },
            "chunks": docs,
        }
    except Exception as e:
        return {
            "text": f"Could not download quarterly report for {symbol}: {e}",
            "sources": {
                "agent": "download_report_agent_tool",
                "symbol": symbol,
                "error": str(e),
                "source_links": [
                    {
                        "label": f"PSX Filings for {symbol}",
                        "url": f"https://dps.psx.com.pk/company/{symbol}",
                    }
                ],
            },
            "chunks": [],
        }


# ============================================================
# SUPERVISOR TOOL DEFINITIONS
# ============================================================

def _symbol_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties={"symbol": types.Schema(type=types.Type.STRING, description="PSX ticker")},
        required=["symbol"],
    )


def _market_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "symbol": types.Schema(type=types.Type.STRING, description="PSX ticker"),
            "query": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The user's original question. Used to infer the analysis "
                    "window (e.g. 'last quarter' → 90 days, 'past year' → 365). "
                    "Pass the full user question verbatim; empty string if unknown."
                ),
            ),
        },
        required=["symbol"],
    )


def _rag_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "symbol": types.Schema(type=types.Type.STRING),
            "query": types.Schema(type=types.Type.STRING),
        },
        required=["symbol"],
    )


supervisor_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="market_data_agent",
        description=(
            "Fetch EOD price/volume for a time window (default 90 days) and a "
            "technical summary. Pass the user's original query in `query` so "
            "phrases like 'last quarter' or 'past year' set the correct window."
        ),
        parameters=_market_schema(),
    ),
    types.FunctionDeclaration(
        name="sentiment_agent",
        description="Summarize recent news sentiment for a ticker.",
        parameters=_symbol_schema(),
    ),
    types.FunctionDeclaration(
        name="risk_agent",
        description="Assess material risks.",
        parameters=_symbol_schema(),
    ),
    types.FunctionDeclaration(
        name="macro_agent",
        description="Fetch GDP, Inflation, and Macroeconomic indicators.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="rag_agent",
        description="Search existing quarterly PDFs in the vector store. Pass the ticker symbol explicitly.",
        parameters=_rag_schema(),
    ),
    types.FunctionDeclaration(
        name="download_report_agent_tool",
        description=(
            "If rag_agent returns no chunks for a ticker, use this. It goes to the PSX "
            "website, finds the latest quarterly PDF, downloads it, OCRs it, and ingests it "
            "into Pinecone. Then re-run rag_agent (or use returned chunks directly)."
        ),
        parameters=_symbol_schema(),
    ),
])

TOOL_MAP = {
    "market_data_agent": market_data_agent,
    "sentiment_agent": sentiment_agent,
    "risk_agent": risk_agent,
    "macro_agent": macro_agent,
    "rag_agent": rag_agent,
    "download_report_agent_tool": download_report_agent_tool,
}

# Agents that spend an LLM call — kept for compatibility, no longer used for throttling
GEMINI_AGENTS = {"market_data_agent", "sentiment_agent", "risk_agent"}

# Tools that must NOT be called multiple times in a single turn
SINGLE_CALL_TOOLS = {"download_report_agent_tool"}


def _run_tool_safe(tool_name: str, args: dict) -> dict:
    """
    Run a tool, capturing exceptions so parallel siblings aren't affected.
    Distinguishes transient errors (retryable) from programmer errors (logged).
    """
    fn = TOOL_MAP.get(tool_name)
    if fn is None:
        return {
            "text": f"Unknown tool: {tool_name}",
            "sources": {"agent": tool_name, "error": f"unknown_tool: {tool_name}",
                        "internal": True, "source_links": []},
            "chunks": [],
        }

    try:
        return fn(**args)
    except (ConnectionError, TimeoutError, OSError, LLMUnavailable) as e:
        return {
            "text": f"Transient error: {e}",
            "sources": {"agent": tool_name, "error": str(e), "retryable": True, "source_links": []},
            "chunks": [],
        }
    except Exception as e:
        print(f"🚨 Tool {tool_name} crashed (programmer error):")
        traceback.print_exc()
        return {
            "text": f"Internal error in {tool_name}",
            "sources": {"agent": tool_name, "error": "internal_error",
                        "internal": True, "source_links": []},
            "chunks": [],
        }


SUPERVISOR_PROMPT = """
ROLE AND IDENTITY
You are the Chief Financial Intelligence Supervisor of a Pakistan Stock Exchange (PSX) market intelligence system.
You coordinate a team of specialized analytical agents and convert their findings into accurate, evidence-based, structured financial intelligence reports.

AVAILABLE SPECIALIST AGENTS
1. market_data_agent: Retrieves EOD price/volume for a time window and a technical summary.
2. sentiment_agent: Analyzes news headlines and classifies sentiment.
3. risk_agent: Performs rule-based checks and identifies risks.
4. macro_agent: Retrieves Pakistani macroeconomic indicators.
5. rag_agent: Searches EXISTING quarterly PDFs in the vector store.
6. download_report_agent_tool: Goes to PSX website, downloads the latest quarterly PDF,
   OCRs it, and ingests it into the vector store. Use ONLY when rag_agent is empty.

TOOL SELECTION AND ORCHESTRATION
Do not automatically call every agent for every question. Select agents according to the user's actual analytical objective.

IMPORTANT: Independent agents run in PARALLEL. If you need multiple independent data sources
(market data + sentiment + risk), call them ALL IN THE SAME TURN. Do NOT stagger them across
multiple turns — that doubles latency.

MARKET DATA WINDOW RULE:
When calling market_data_agent, ALWAYS pass the user's original question in
the `query` field verbatim. The agent will infer the time window from
phrases like "last month" (30 days), "last quarter" (90 days), "past year"
(365 days), or a specific number like "180 days". If the user said nothing
about time, pass the query anyway — the agent will default to 90 days.

QUARTERLY REPORT FALLBACK RULE (IMPORTANT):
1. Call rag_agent first.
2. If rag_agent returns "No relevant PDF documents found", then call download_report_agent_tool.
3. After download_report_agent_tool completes, either use its returned chunks directly
   OR call rag_agent one more time. Do NOT tell the user to "try again later" —
   the download is synchronous and happens during this request.
4. If download_report_agent_tool fails, note the failure in the report's "Analytical Limitations" section.

EVIDENCE AND DATA VALIDATION
1. Never fabricate stock prices, trading volumes, financial ratios, or news events.
2. Treat tool outputs as evidence, not unquestionable truth.
3. Where information is missing, explicitly identify the missing evidence.
4. If different agents provide conflicting findings, identify the disagreement.

🚫 CRITICAL FORMATTING RULE:
Do NOT use Markdown tables (no pipe characters '|', no dash separators '---').
Present all financial data using bullet points, numbered lists, or structured paragraphs.
NEVER format your output as a table.

FINAL REPORT FORMAT
# PSX Financial Intelligence Report (200-350 words and use simple plain english)
## 1. Executive Summary
## 2. Market Performance
## 3. News and Sentiment
## 4. Macroeconomic Environment
## 5. Quarterly Financial Intelligence
## 6. Risk Assessment
## 7. Integrated Intelligence
## 8. Analytical Limitations
## 9. Overall Assessment

VERDICT DISCIPLINE (IMPORTANT):

Base BUY/SELL/HOLD on the strongest evidence, not the weakest risk.

If ALL of the following are true:
  • Quarterly EPS or profit has grown ≥20% YoY
  • RSI is between 40 and 65 (not overbought)
  • Volume ratio is ≥ 1.5× the 30-day average
  • Price is above the 20-day Bollinger Mid

Then VERDICT MUST be BUY, with HOLDING_PERIOD 15-30 days.
Do NOT default to HOLD just because risks exist. Every stock has risks.

The reverse applies: if profit is falling, price is below Bollinger Mid,
and RSI > 65, VERDICT MUST be SELL. Do NOT default to HOLD just because
some upside could exist.

HOLD is only correct when the signals genuinely conflict
(e.g., strong earnings but deeply negative MACD and RSI < 35).

If you describe a breakout setup in the body, the verdict MUST be BUY.
Your prose and your verdict cannot contradict each other.


our report MUST end with exactly these three lines (no markdown, no asterisks, no bold):
VERDICT: BUY
CONFIDENCE: 0.75
HOLDING_PERIOD: 15 days

Replace "BUY" with "SELL" or "HOLD" based on your analysis.
Replace "0.75" with your actual confidence as a decimal between 0.0 and 1.0.
Replace "15" with the number of TRADING DAYS you recommend acting on this signal (1-90):
  - For BUY: how long to hold before taking profit or re-evaluating.
  - For SELL: how long to wait before re-entering (or 0 for immediate exit).
  - For HOLD: how long before re-checking the position.
Do NOT add "Prepared by", "Disclaimer", or any text after these three lines.
Provide Warning about This is for educational purposes only. This is not financial advice. Do not use this information to make investment decisions.
"""


# ============================================================
# GRAPH
# ============================================================

class PSXAgentGraph:
    def invoke(self, state: Dict[str, Any]) -> Dict[str, Any]:
        final_state = None
        for event in self.stream(state):
            if "__end__" in event:
                final_state = event["__end__"]
        if not final_state:
            return {"final_report": "Error: Graph execution failed.", "data_sources": {}}
        out = dict(state)
        out.update(final_state)
        return out

    def stream(
        self, state: Dict[str, Any], stream_mode: str = "updates"
    ) -> Generator[Dict[str, Any], None, None]:
        query = state.get("query", "Market overview")
        symbol = (state.get("symbol") or "").upper() or extract_symbol_from_query(query)

        contents = [types.Content(
            role="user",
            parts=[types.Part(text=f"Ticker: {symbol}. Request: {query}")],
        )]
        config = types.GenerateContentConfig(
            system_instruction=SUPERVISOR_PROMPT,
            tools=[supervisor_tool],
            temperature=0.2,
        )

        data_sources: Dict[str, Any] = {}
        numerical_data: Dict[str, Any] = {}
        risk_alerts: List[str] = []
        retrieved_docs: List[str] = []
        sentiment_summary: str = ""

        supervisor_crashed = False
        download_already_called = False
        consecutive_failures = 0

        for _step in range(6):
            # Only sleep after a failed round — no unconditional delay
            if _step > 0 and consecutive_failures > 0:
                time.sleep(min(3, consecutive_failures))

            try:
                resp = gemini_client.models.generate_content(
                    model=MODEL, contents=contents, config=config
                )
            except Exception as api_err:
                print(f"⚠️ Supervisor crashed: {api_err}. Running manual fallback...")
                supervisor_crashed = True
                break

            # Guard against empty candidates (Gemini can return this on safety block)
            if not resp.candidates or not resp.candidates[0].content:
                print("⚠️ Supervisor returned no candidates. Running manual fallback...")
                supervisor_crashed = True
                break

            contents.append(resp.candidates[0].content)

            # No more tools → final report
            if not resp.function_calls:
                final_text = (resp.text or "").strip() or "⚠️ Supervisor returned empty report."
                yield {
                    "__end__": {
                        "symbol": symbol,
                        "final_report": final_text,
                        "numerical_data": numerical_data,
                        "risk_alerts": risk_alerts,
                        "retrieved_docs": retrieved_docs,
                        "sentiment_summary": sentiment_summary,
                        "data_sources": data_sources,
                    }
                }
                return

            # ==================================================
            # PARALLEL tool execution
            # ==================================================
            tool_parts = []
            pre_handled_parts = []
            tasks: List[tuple] = []

            # Phase A — classify tools & build task list
            for fc in resp.function_calls:
                args = dict(fc.args)

                # Guard: skip duplicate download calls
                if fc.name in SINGLE_CALL_TOOLS and download_already_called:
                    pre_handled_parts.append(types.Part.from_function_response(
                        name=fc.name,
                        response={"result": "Already called. Use the chunks you received from the first call."},
                    ))
                    continue

                # Flip the flag BEFORE submitting, so a second call in the
                # same batch is also rejected.
                if fc.name == "download_report_agent_tool":
                    yield {"download_report_agent_tool": {"status": "running", "symbol": args.get("symbol")}}
                    download_already_called = True

                tasks.append((fc, args))

            # Phase B — run all real tasks in parallel
            results_by_idx: Dict[int, tuple] = {}
            round_had_failure = False
            if tasks:
                max_workers = min(len(tasks), MAX_PARALLEL_TOOLS)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_info = {}
                    for idx, (fc, args) in enumerate(tasks):
                        future = executor.submit(_run_tool_safe, fc.name, args)
                        future_to_info[future] = (idx, fc)

                    for future in as_completed(future_to_info):
                        idx, fc = future_to_info[future]
                        result = future.result()      # _run_tool_safe never raises
                        results_by_idx[idx] = (fc, result)
                        if isinstance(result, dict) and (
                            result.get("sources", {}).get("error")
                            or result.get("sources", {}).get("internal")
                        ):
                            round_had_failure = True
                        yield {fc.name: {"status": "completed"}}

            consecutive_failures = (consecutive_failures + 1) if round_had_failure else 0

            # Phase C — update shared state in submission order (deterministic)
            for idx in range(len(tasks)):
                fc, result = results_by_idx[idx]

                text = result.get("text", "") if isinstance(result, dict) else str(result)
                sources = result.get("sources", {}) if isinstance(result, dict) else {}
                chunks = result.get("chunks", []) if isinstance(result, dict) else []

                data_sources[fc.name] = sources

                if fc.name == "market_data_agent":
                    numerical_data = sources.get("latest_row", {}) or {}
                elif fc.name == "risk_agent":
                    risk_alerts = sources.get("alerts", []) or []
                elif fc.name in ("rag_agent", "download_report_agent_tool"):
                    if chunks:
                        retrieved_docs = chunks
                elif fc.name == "sentiment_agent":
                    sentiment_summary = text

                tool_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": text}
                ))

            # Prepend "already called" responses
            tool_parts = pre_handled_parts + tool_parts

            contents.append(types.Content(role="user", parts=tool_parts))

        # ---------- FALLBACK: supervisor crashed or exceeded steps ----------
        if supervisor_crashed or not retrieved_docs:
            print("🔄 Running manual fallback chain...")

            try:
                yield {"market_data_agent": {"status": "completed"}}
                m_res = market_data_agent(symbol, query=query)
                data_sources["market_data_agent"] = m_res.get("sources", {})
                numerical_data = m_res.get("sources", {}).get("latest_row", {}) or {}

                yield {"sentiment_agent": {"status": "completed"}}
                news_res = sentiment_agent(symbol)
                data_sources["sentiment_agent"] = news_res.get("sources", {})
                sentiment_summary = news_res.get("text", "")

                yield {"rag_agent": {"status": "completed"}}
                rag_res = rag_agent(symbol)
                data_sources["rag_agent"] = rag_res.get("sources", {})
                retrieved_docs = rag_res.get("chunks", [])

                if not retrieved_docs and not download_already_called:
                    yield {"download_report_agent_tool": {"status": "running", "symbol": symbol}}
                    dl_res = download_report_agent_tool(symbol)
                    data_sources["download_report_agent_tool"] = dl_res.get("sources", {})
                    retrieved_docs = dl_res.get("chunks", [])
                    yield {"download_report_agent_tool": {"status": "completed"}}

                yield {"risk_agent": {"status": "completed"}}
                risk_res = risk_agent(symbol)
                data_sources["risk_agent"] = risk_res.get("sources", {})
                risk_alerts = risk_res.get("sources", {}).get("alerts", [])

                yield {"macro_agent": {"status": "completed"}}
                macro_res = macro_agent()
                data_sources["macro_agent"] = macro_res.get("sources", {})

            except Exception as e:
                print(f"⚠️ Error running manual tools: {e}")

            fallback_prompt = f"""
You are the Chief Financial Intelligence Supervisor of a Pakistan Stock Exchange (PSX) market intelligence system.
The primary AI supervisor crashed, but I have manually gathered the following raw data for {symbol}:


Strict Rule:
The quaterly report ingestion must be exected first as it takes time to download it from the webstie


**Market Data:** {numerical_data}
**Risk Alerts:** {risk_alerts}
**Sentiment Summary:** {sentiment_summary}
**Financial Docs (quarterly report excerpts):** {retrieved_docs}

EVIDENCE AND DATA VALIDATION
1. Never fabricate stock prices, trading volumes, financial ratios, or news events.
2. Treat tool outputs as evidence, not unquestionable truth.
3. Where information is missing, explicitly identify the missing evidence.
4. If different agents provide conflicting findings, identify the disagreement.

🚫 CRITICAL FORMATTING RULE:
Do NOT use Markdown tables (no pipe characters '|', no dash separators '---').
Present all financial data using bullet points, numbered lists, or structured paragraphs.

FINAL REPORT FORMAT [200-350 words total and use simple plain english]
# PSX Financial Intelligence Report
## 1. Executive Summary 
## 2. Market Performance
## 3. News and Sentiment
## 4. Macroeconomic Environment
## 5. Quarterly Financial Intelligence
## 6. Risk Assessment
## 7. Integrated Intelligence
## 8. Analytical Limitations (3 concsie simple bullet points)
## 9. Overall Assessment

our report MUST end with exactly these three lines (no markdown, no asterisks, no bold):
VERDICT: BUY
CONFIDENCE: 0.75
HOLDING_PERIOD: 15 days

Replace "BUY" with "SELL" or "HOLD" based on your analysis.
Replace "0.75" with your actual confidence as a decimal between 0.0 and 1.0.
Replace "15" with the number of TRADING DAYS you recommend acting on this signal (1-90):
  - For BUY: how long to hold before taking profit or re-evaluating.
  - For SELL: how long to wait before re-entering (or 0 for immediate exit).
  - For HOLD: how long before re-checking the position.

  
Do NOT add "Prepared by", "Disclaimer", or any text after these three lines.
"""
            try:
                final_report = generate_text(fallback_prompt, "You are a professional financial analyst.")
            except LLMUnavailable as e:
                print(f"⚠️ Fallback synthesis skipped (all LLMs down): {e}")
                final_report = (
                    f"⚠️ Supervisor and LLM providers both unavailable.\n"
                    f"Raw data was collected for {symbol} but could not be synthesized.\n"
                    f"Market data: {numerical_data}\nRisk alerts: {risk_alerts}"
                )

            yield {
                "__end__": {
                    "symbol": symbol,
                    "final_report": final_report,
                    "numerical_data": numerical_data,
                    "risk_alerts": risk_alerts,
                    "retrieved_docs": retrieved_docs,
                    "sentiment_summary": sentiment_summary,
                    "data_sources": data_sources,
                }
            }
            return

        # ---------- Reached max steps without a final answer ----------
        yield {
            "__end__": {
                "symbol": symbol,
                "final_report": "⚠️ Supervisor exceeded max steps.",
                "numerical_data": numerical_data,
                "risk_alerts": risk_alerts,
                "retrieved_docs": retrieved_docs,
                "sentiment_summary": sentiment_summary,
                "data_sources": data_sources,
            }
        }


def build_psx_graph() -> PSXAgentGraph:
    return PSXAgentGraph()