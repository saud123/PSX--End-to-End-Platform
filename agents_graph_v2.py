"""
PSX Intelligence — v2: SINGLE SUPERVISOR WITH RAW DATA TOOLS

Changes carried over from V1:
  • Safe-hybrid market window: LLM extracts N days, SQL template is fixed.
  • Synchronous download tool for quarterly PDFs (no background threads).
  • Parallel tool execution — independent tools run concurrently.
"""

import os
import time
import json
from datetime import datetime
from typing import TypedDict, List, Dict, Any, Generator
from concurrent.futures import ThreadPoolExecutor, as_completed

from google.genai import types
from sqlalchemy import text

from llm import generate_text  # for fallback synthesis
from agents_graph import (
    gemini_client, MODEL, engine, fetch_market_data, fetch_news,
    fetch_rag_matches, rule_based_risk_checks, extract_symbol_from_query,
    NEWS_SOURCE_HOME, fetch_multi_query_rag,
    # ✅ Safe-hybrid market window helpers (from V1)
    _parse_days_from_query, _fetch_market_window, DEFAULT_MARKET_DAYS,
)

# Parallelism cap — keeps Gemini/DB/Pinecone from being hammered
MAX_PARALLEL_TOOLS = 6


# ============================================================
# RAW DATA TOOLS
# ============================================================

def tool_get_price_data(symbol: str, query: str = "") -> Dict[str, Any]:
    """
    Fetch a sliding window of EOD prices.
    N days is inferred from `query` by the LLM (safe — output is an integer),
    and the SQL template is hardcoded in agents_graph._fetch_market_window.
    Default 90 days.
    """
    days = _parse_days_from_query(query) if query else DEFAULT_MARKET_DAYS
    window = _fetch_market_window(symbol, days)

    return {
        "symbol": symbol,
        "days": days,
        "count": window["count"],
        "latest_row": window["latest_row"],
        "history": window["history"],     # full window for the supervisor to reason over
        "source_links": [
            {"label": "PSX Portal", "url": f"https://dps.psx.com.pk/company/{symbol}"}
        ],
    }


def tool_get_news(symbol: str, limit: int = 8) -> Dict[str, Any]:
    articles = fetch_news(symbol, limit)
    seen_sources: List[str] = []
    for a in articles:
        s = a.get("source") or ""
        if s and s not in seen_sources:
            seen_sources.append(s)

    return {
        "symbol": symbol,
        "articles": [
            {
                "headline": (a.get("headline") or "")[:180],
                "source": a.get("source") or "",
                "label": a.get("sentiment") or "Neutral",
                "url": a.get("url", ""),
            }
            for a in articles
        ],
        "news_count": len(articles),
        "source_links": [
            {"label": s, "url": NEWS_SOURCE_HOME[s]}
            for s in seen_sources if s in NEWS_SOURCE_HOME
        ],
    }


def tool_get_macro() -> Dict[str, Any]:
    try:
        with engine.connect() as conn:
            res = conn.execute(text(
                "SELECT page_category, chart_title, jsonb_strip_nulls(raw_data) as data "
                "FROM finance_macro_dashboard ORDER BY id DESC LIMIT 20"
            ))
            rows = [dict(r._mapping) for r in res]
        seen, charts = set(), []
        for row in rows:
            key = (row["page_category"], row["chart_title"])
            if key in seen:
                continue
            seen.add(key)
            charts.append({
                "category": row["page_category"].replace("-", " ").title(),
                "chart": row["chart_title"],
                "data": row["data"],
            })
        return {
            "charts": charts,
            "charts_count": len(charts),
            "source_links": [{"label": "Ministry of Finance", "url": "https://economy.finance.gov.pk"}],
        }
    except Exception as e:
        return {"charts": [], "charts_count": 0, "error": str(e), "source_links": []}


def tool_search_reports(symbol: str, query: str = "") -> Dict[str, Any]:
    """
    Search Pinecone only. NO background ingestion.
    If empty, return `empty: True` and let the supervisor call `download_report`.
    """
    docs = fetch_multi_query_rag(symbol)
    matches = [{"text": doc} for doc in docs]

    if not matches:
        return {
            "symbol": symbol,
            "chunks": [],
            "chunks_count": 0,
            "empty": True,
            "source_links": [
                {"label": f"PSX Filings for {symbol}", "url": f"https://dps.psx.com.pk/company/{symbol}"}
            ],
        }

    source_links = []
    for m in fetch_rag_matches(f"{symbol} quarterly report", symbol):
        if (m.get("source_url") or "").startswith("http"):
            source_links.append({
                "label": f"PSX Report PDF · {m.get('quarter', '')} {m.get('year', '')} (Page {m.get('page_number', '?')})",
                "url": m["source_url"],
            })
            break
    if not source_links:
        source_links.append({
            "label": f"PSX Filings for {symbol}",
            "url": f"https://dps.psx.com.pk/company/{symbol}",
        })

    return {
        "symbol": symbol,
        "chunks": matches,
        "chunks_count": len(matches),
        "empty": False,
        "source_links": source_links,
    }


def tool_download_report(symbol: str) -> Dict[str, Any]:
    """
    THE SINGLE DOWNLOAD AGENT (as a tool):
    PSX → find PDF → download → OCR → chunk → embed → upsert.
    Synchronous. Returns chunks ready for the supervisor to use.

    Handles two outcomes (matching the new ingestion_report contract):
      • pdf_ingested=True  → PDF chunks + metrics ingested
      • pdf_ingested=False → only metrics ingested
    """
    try:
        from ingestion_report import download_report_agent

        result = download_report_agent(symbol)
        docs = fetch_multi_query_rag(symbol)

        return {
            "symbol": symbol,
            "pdf_ingested": bool(result.get("pdf_ingested")),
            "metrics_ingested": result.get("metrics_ingested", 0),
            "quarter": result.get("quarter"),
            "year": result.get("year"),
            "pages": result.get("pages"),
            "ocr_pages": result.get("ocr_pages"),
            "ingested_chunks": result.get("chunks", 0),
            "chunks_count": len(docs),
            "chunks": [{"text": d} for d in docs],
            "reason": result.get("reason"),
            "source_links": [
                {
                    "label": f"PSX company page ({symbol})",
                    "url": f"https://dps.psx.com.pk/company/{symbol}",
                }
            ],
        }
    except Exception as e:
        return {
            "symbol": symbol,
            "error": str(e),
            "chunks": [],
            "chunks_count": 0,
            "source_links": [
                {
                    "label": f"PSX Filings for {symbol}",
                    "url": f"https://dps.psx.com.pk/company/{symbol}",
                }
            ],
        }


def tool_get_risk_flags(symbol: str) -> Dict[str, Any]:
    market_data = fetch_market_data(symbol)
    alerts = rule_based_risk_checks(symbol, market_data)
    return {"symbol": symbol, "alerts": alerts, "source_links": []}


# ============================================================
# TOOL SCHEMAS
# ============================================================

def _sym_schema() -> types.Schema:
    return types.Schema(
        type=types.Type.OBJECT,
        properties={"symbol": types.Schema(type=types.Type.STRING)},
        required=["symbol"],
    )


def _price_schema() -> types.Schema:
    """get_price_data — adds optional query for time-window inference."""
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "symbol": types.Schema(type=types.Type.STRING),
            "query": types.Schema(
                type=types.Type.STRING,
                description=(
                    "The user's original question verbatim. Used to infer the "
                    "analysis window ('last quarter' → 90 days, 'past year' → 365). "
                    "Pass the full question; empty string if unknown."
                ),
            ),
        },
        required=["symbol"],
    )


v2_tool = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="get_price_data",
        description=(
            "Fetch EOD price data for a time window (default 90 days). "
            "Pass the user's original question in `query` so phrases like "
            "'last quarter' or 'past year' set the correct window."
        ),
        parameters=_price_schema(),
    ),
    types.FunctionDeclaration(
        name="get_news",
        description="Fetch up to 8 recent news headlines.",
        parameters=_sym_schema(),
    ),
    types.FunctionDeclaration(
        name="get_macro",
        description="Fetch macroeconomic charts.",
        parameters=types.Schema(type=types.Type.OBJECT, properties={}),
    ),
    types.FunctionDeclaration(
        name="search_reports",
        description="Search existing quarterly PDF reports in the vector store.",
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "symbol": types.Schema(type=types.Type.STRING),
                "query": types.Schema(type=types.Type.STRING),
            },
            required=["symbol"],
        ),
    ),
    types.FunctionDeclaration(
        name="download_report",
        description=(
            "If search_reports returns no chunks, use this. Goes to PSX website, "
            "downloads the latest quarterly PDF, OCRs it, and ingests it into Pinecone. "
            "Then use the returned chunks directly."
        ),
        parameters=_sym_schema(),
    ),
    types.FunctionDeclaration(
        name="get_risk_flags",
        description="Get deterministic rule-based risk alerts.",
        parameters=_sym_schema(),
    ),
])

V2_TOOL_MAP = {
    "get_price_data": tool_get_price_data,
    "get_news": tool_get_news,
    "get_macro": tool_get_macro,
    "search_reports": tool_search_reports,
    "download_report": tool_download_report,
    "get_risk_flags": tool_get_risk_flags,
}

V2_TOOL_TO_SLOT = {
    "get_price_data": "market_data_agent",
    "get_news": "sentiment_agent",
    "search_reports": "rag_agent",
    "download_report": "download_report_agent_tool",
    "get_risk_flags": "risk_agent",
    "get_macro": "macro_agent",
}

# Tools that must not be called twice in one turn
V2_SINGLE_CALL_TOOLS = {"download_report"}


def _run_v2_tool_safe(tool_name: str, args: dict) -> dict:
    """
    Run a V2 raw-data tool, capturing any exception so parallel siblings
    aren't affected. Always returns a dict so downstream processing is
    uniform.
    """
    try:
        fn = V2_TOOL_MAP.get(tool_name)
        if fn is None:
            raise KeyError(f"Unknown tool: {tool_name}")
        return fn(**args)
    except Exception as e:
        return {"error": str(e), "source_links": []}


V2_PROMPT = """
# ROLE: CHIEF FINANCIAL INTELLIGENCE ANALYST
# ARCHITECTURE: SINGLE SUPERVISOR + RAW DATA TOOLS
You operate through a SINGLE SUPERVISOR architecture.
You directly access raw financial information through specialized tools and independently perform financial market analysis, news sentiment classification, macroeconomic interpretation, quarterly financial report analysis, risk identification, and financial intelligence reporting.
Accuracy must always take priority over completeness.

# AVAILABLE RAW DATA TOOLS
TOOL 1: get_price_data (EOD price data for a time window — default 90 days)
TOOL 2: get_news (Headlines)
TOOL 3: get_macro (Macro indicators)
TOOL 4: search_reports (PDF chunks from the vector store)
TOOL 5: download_report (Download latest PDF from PSX and ingest into the vector store)
TOOL 6: get_risk_flags (Rule alerts)

# INTELLIGENT TOOL SELECTION
Do not call every tool. Select tools based on the request.

IMPORTANT: Independent tools run in PARALLEL. If you need multiple independent data
sources (price + news + risk), call them ALL IN THE SAME TURN. Do NOT stagger them
across multiple turns — that doubles latency.

# MARKET DATA WINDOW RULE
When calling get_price_data, ALWAYS pass the user's original question in the
`query` field verbatim. The tool infers the time window from phrases like
"last month" (30 days), "last quarter" (90 days), "past year" (365 days),
or a specific number like "180 days". If the user said nothing about time,
still pass the query — the tool will default to 90 days.

# QUARTERLY REPORT FALLBACK RULE (IMPORTANT)
1. Call search_reports first.
2. If it returns no chunks (empty=true or chunks_count=0), call download_report.
3. After download_report completes, use its returned chunks directly.
   Do NOT tell the user to "try again later" — the download is synchronous.
4. If download_report fails, note it under "Data Quality and Limitations".

🚫 CRITICAL FORMATTING RULE:
Do NOT use Markdown tables (no pipe characters '|', no dash separators '---').
Present all financial data using bullet points, numbered lists, or structured paragraphs.
NEVER format your output as a table.

# FINAL REPORT FORMAT
# PSX Financial Intelligence Report [200-350 words and use simple plain English]
## 1. Executive Summary
## 2. Market Performance
## 3. News and Sentiment Analysis
## 4. Macroeconomic Environment
## 5. Quarterly Financial Analysis
## 6. Risk Assessment
## 7. Cross-Source Intelligence
## 8. Data Quality and Limitations
## 9. Overall Assessment
## 10. Your report MUST end with exactly these three lines (no markdown):
VERDICT: BUY
CONFIDENCE: 0.75
HOLDING_PERIOD: 15 days
Replace BUY with SELL or HOLD. Replace 0.75 with your confidence (0.0-1.0).
Replace 15 with recommended holding days in trading days (1-90). For SELL, 0 = exit immediately.
Do NOT add "Prepared by" or "Disclaimer" after these lines.

NEVER SACRIFICE ACCURACY FOR COMPLETENESS.

Provide Warning about This is for educational purposes only. This is not financial advice. Do not use this information to make investment decisions.
"""


# ============================================================
# GRAPH
# ============================================================

class PSXAgentGraphV2:
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
            system_instruction=V2_PROMPT,
            tools=[v2_tool],
            temperature=0.2,
        )

        data_sources: Dict[str, Any] = {}
        numerical_data: Dict[str, Any] = {}
        risk_alerts: List[str] = []
        retrieved_docs: List[Dict[str, Any]] = []
        sentiment_summary: str = ""
        market_window: Dict[str, Any] = {}

        supervisor_failed = False
        download_already_called = False

        for _step in range(6):
            if _step > 0:
                time.sleep(3)

            try:
                resp = gemini_client.models.generate_content(
                    model=MODEL, contents=contents, config=config
                )
            except Exception as api_err:
                print(f"⚠️ v2 Supervisor crashed: {api_err}. Running manual fallback...")
                supervisor_failed = True
                break

            if not resp.candidates or not resp.candidates[0].content:
                yield {
                    "__end__": {
                        "symbol": symbol,
                        "final_report": "⚠️ Model returned no content.",
                        "numerical_data": numerical_data,
                        "risk_alerts": risk_alerts,
                        "retrieved_docs": retrieved_docs,
                        "sentiment_summary": sentiment_summary,
                        "data_sources": data_sources,
                        "version": "v2",
                    }
                }
                return

            contents.append(resp.candidates[0].content)

            if not resp.function_calls:
                yield {
                    "__end__": {
                        "symbol": symbol,
                        "final_report": (resp.text or "").strip() or "(model returned no text)",
                        "numerical_data": numerical_data,
                        "risk_alerts": risk_alerts,
                        "retrieved_docs": retrieved_docs,
                        "sentiment_summary": sentiment_summary,
                        "data_sources": data_sources,
                        "version": "v2",
                    }
                }
                return

            # ==================================================
            # PARALLEL tool execution (main change)
            # ==================================================
            tool_parts = []
            pre_handled_parts = []
            tasks: List[tuple] = []

            # Phase A — classify tools, handle duplicates, emit "running"
            for fc in resp.function_calls:
                args = dict(fc.args)

                # Guard: skip duplicate download calls
                if fc.name in V2_SINGLE_CALL_TOOLS and download_already_called:
                    pre_handled_parts.append(types.Part.from_function_response(
                        name=fc.name,
                        response={"result": "Already called. Use the chunks you received from the first call."},
                    ))
                    continue

                # Flip the flag BEFORE submission so a second call in the
                # same batch is also rejected.
                if fc.name == "download_report":
                    yield {"download_report": {"status": "running", "symbol": args.get("symbol")}}
                    download_already_called = True

                tasks.append((fc, args))

            # Phase B — run all real tasks in parallel
            results_by_idx: Dict[int, tuple] = {}
            if tasks:
                max_workers = min(len(tasks), MAX_PARALLEL_TOOLS)
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_info = {}
                    for idx, (fc, args) in enumerate(tasks):
                        future = executor.submit(_run_v2_tool_safe, fc.name, args)
                        future_to_info[future] = (idx, fc)

                    for future in as_completed(future_to_info):
                        idx, fc = future_to_info[future]
                        raw = future.result()          # _run_v2_tool_safe never raises
                        results_by_idx[idx] = (fc, raw)
                        yield {fc.name: {"status": "completed"}}

            # Phase C — update shared state in submission order (deterministic)
            for idx in range(len(tasks)):
                fc, raw = results_by_idx[idx]
                slot = V2_TOOL_TO_SLOT.get(fc.name, fc.name)

                if fc.name == "get_price_data":
                    numerical_data = raw.get("latest_row", {}) or {}
                    market_window = raw
                elif fc.name == "get_risk_flags":
                    risk_alerts = raw.get("alerts", []) or []
                elif fc.name in ("search_reports", "download_report"):
                    if raw.get("chunks"):
                        retrieved_docs = raw["chunks"]

                data_sources[slot] = {
                    "agent": slot,
                    "mode": "raw_tool",
                    "tool_name": fc.name,
                    **{k: v for k, v in raw.items() if k != "chunks"},
                }

                text_summary = self._summarize_for_llm(fc.name, raw)
                tool_parts.append(types.Part.from_function_response(
                    name=fc.name, response={"result": text_summary}
                ))

            # Prepend "already called" responses (rare, keeps contract clean)
            tool_parts = pre_handled_parts + tool_parts

            contents.append(types.Content(role="user", parts=tool_parts))

        # ---------- FALLBACK: supervisor crashed or exceeded steps (SEQUENTIAL) ----------
        if supervisor_failed or not retrieved_docs:
            print("🔄 v2 Supervisor failed. Running manual fallback...")

            try:
                # 1. Price data (with query → days inference)
                yield {"get_price_data": {"status": "completed"}}
                m_data = tool_get_price_data(symbol, query=query)
                numerical_data = m_data.get("latest_row", {})
                market_window = m_data
                data_sources["market_data_agent"] = m_data

                # 2. News
                yield {"get_news": {"status": "completed"}}
                news_res = tool_get_news(symbol)
                data_sources["sentiment_agent"] = news_res
                news_articles = news_res.get("articles", [])

                # 3. RAG first try
                yield {"search_reports": {"status": "completed"}}
                rag_res = tool_search_reports(symbol)
                data_sources["rag_agent"] = rag_res
                retrieved_docs = rag_res.get("chunks", [])

                # 3b. Empty → download_report (synchronous)
                if not retrieved_docs and not download_already_called:
                    yield {"download_report": {"status": "running", "symbol": symbol}}
                    dl_res = tool_download_report(symbol)
                    data_sources["download_report_agent_tool"] = dl_res
                    retrieved_docs = dl_res.get("chunks", [])
                    yield {"download_report": {"status": "completed"}}

                # 4. Risk flags
                yield {"get_risk_flags": {"status": "completed"}}
                risk_res = tool_get_risk_flags(symbol)
                data_sources["risk_agent"] = risk_res
                risk_alerts = risk_res.get("alerts", [])

                # 5. Macro
                yield {"get_macro": {"status": "completed"}}
                macro_res = tool_get_macro()
                data_sources["macro_agent"] = macro_res

            except Exception as e:
                print(f"⚠️ Error running manual tools in V2 fallback: {e}")

            fallback_prompt = f"""
The primary AI supervisor crashed, but I have manually gathered the following raw data for {symbol}:

**Market Data (window):** {market_window}
**Risk Alerts:** {risk_alerts}
**News Articles:** {news_articles}
**Financial Docs:** {retrieved_docs}

# ROLE: CHIEF FINANCIAL INTELLIGENCE ANALYST

🚫 CRITICAL FORMATTING RULE: Do NOT use Markdown tables (no pipe characters '|').
Present all financial data using bullet points, numbered lists, or structured paragraphs.

# FINAL REPORT FORMAT
# PSX Financial Intelligence Report [200-350 words and use simple english]
## 1. Executive Summary
## 2. Market Performance
## 3. News and Sentiment Analysis
## 4. Macroeconomic Environment
## 5. Quarterly Financial Analysis
## 6. Risk Assessment
## 7. Cross-Source Intelligence
## 8. Data Quality and Limitations
## 9. Overall Assessment

Your report MUST end with exactly these three lines (no markdown):
VERDICT: BUY
CONFIDENCE: 0.75
HOLDING_PERIOD: 15 days
Replace BUY with SELL or HOLD. Replace 0.75 with your confidence (0.0-1.0). Replace 15 with holding days (1-90).
Do NOT add "Prepared by" or "Disclaimer" after these lines.

Write a comprehensive report based on this data. Do not invent data.
"""
            final_report = generate_text(fallback_prompt, "You are a professional financial analyst.")
            yield {
                "__end__": {
                    "symbol": symbol,
                    "final_report": final_report,
                    "numerical_data": numerical_data,
                    "risk_alerts": risk_alerts,
                    "retrieved_docs": retrieved_docs,
                    "sentiment_summary": sentiment_summary,
                    "data_sources": data_sources,
                    "version": "v2",
                }
            }
            return

        # ---------- Reached max steps ----------
        yield {
            "__end__": {
                "symbol": symbol,
                "final_report": "⚠️ v2 supervisor exceeded max steps.",
                "numerical_data": numerical_data,
                "risk_alerts": risk_alerts,
                "retrieved_docs": retrieved_docs,
                "sentiment_summary": sentiment_summary,
                "data_sources": data_sources,
                "version": "v2",
            }
        }

    # ------------------------------------------------------------
    # Compact text summary for the supervisor's function responses
    # ------------------------------------------------------------
    def _summarize_for_llm(self, tool_name: str, raw: Dict[str, Any]) -> str:
        if tool_name == "get_price_data":
            latest = raw.get("latest_row", {}) or {}
            if not latest:
                return f"No price data for {raw.get('symbol')}."

            days = raw.get("days", DEFAULT_MARKET_DAYS)
            count = raw.get("count", 0)
            history = raw.get("history", []) or []

            closes = [float(h["close"]) for h in history if h.get("close") is not None]
            highs = [float(h["high"]) for h in history if h.get("high") is not None]
            lows = [float(h["low"]) for h in history if h.get("low") is not None]
            vols = [int(h["volume"]) for h in history if h.get("volume") is not None]

            lines = [
                f"{raw.get('symbol')} · last {days} days · {count} rows",
                f"  latest: {latest.get('trade_date')} · close={latest.get('close')} · volume={latest.get('volume')}",
            ]
            if closes:
                lines.append(
                    f"  window: high={max(highs):.2f} · low={min(lows):.2f} · "
                    f"avg_close={sum(closes) / len(closes):.2f}"
                )
            if vols:
                lines.append(f"  avg_volume={int(sum(vols) / len(vols)):,}")
            if len(closes) >= 2 and closes[0] > 0:
                pct = (closes[-1] - closes[0]) / closes[0] * 100
                lines.append(f"  pct_change_over_window={pct:+.2f}%")
            return "\n".join(lines)

        if tool_name == "get_news":
            arts = raw.get("articles", [])
            if not arts:
                return f"No recent news for {raw.get('symbol')}."
            return "\n".join(
                f"- {a['headline']} ({a.get('source', '')})" for a in arts[:8]
            )

        if tool_name == "get_macro":
            charts = raw.get("charts", [])
            if not charts:
                return "No macro data available."
            return "\n".join(
                f"- {c['category']} / {c['chart']}: {c['data']}" for c in charts
            )

        if tool_name == "search_reports":
            chunks = raw.get("chunks", [])
            if not chunks:
                return "No quarterly report chunks retrieved. (Consider calling download_report.)"
            body = "\n\n".join(c.get("text", "") for c in chunks[:3])
            return f"{len(chunks)} chunks retrieved:\n\n{body}"

        # ---------- 1c: honest summary based on pdf_ingested / metrics_ingested ----------
        if tool_name == "download_report":
            if raw.get("error"):
                return f"Download failed for {raw.get('symbol')}: {raw['error']}"

            chunks = raw.get("chunks", [])

            if raw.get("pdf_ingested"):
                header = (
                    f"Downloaded + ingested {raw.get('quarter')} {raw.get('year')} PDF "
                    f"for {raw.get('symbol')} "
                    f"({raw.get('pages')} pages, {raw.get('ocr_pages')} OCR'd). "
                    f"Vector store now has {raw.get('chunks_count')} chunks."
                )
            elif int(raw.get("metrics_ingested", 0) or 0) > 0:
                header = (
                    f"No fresh quarterly PDF for {raw.get('symbol')} "
                    f"(reason: {raw.get('reason', 'none')}). "
                    f"Ingested {raw['metrics_ingested']} metrics doc(s) "
                    f"(Financials + Ratios). "
                    f"Vector store now has {raw.get('chunks_count')} chunks."
                )
            else:
                return f"No PDF or metrics ingested for {raw.get('symbol')}."

            if not chunks:
                return header
            body = "\n\n".join(c.get("text", "")[:800] for c in chunks[:3])
            return f"{header}\n\n{body}"

        if tool_name == "get_risk_flags":
            alerts = raw.get("alerts", [])
            return f"Rule alerts ({len(alerts)}): " + ("; ".join(alerts) if alerts else "none")

        return json.dumps(raw, default=str)[:1500]


def build_psx_graph() -> PSXAgentGraphV2:
    return PSXAgentGraphV2()