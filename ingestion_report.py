# ingestion_report.py — Level 1 agentic extraction
# Change vs. previous version:
#   The four hard-coded extractors + heuristic + LLM fallback are replaced by ONE
#   LLM call (_agentic_extract) that reads the page and returns:
#     - which quarterly PDF to download (with a reason and confidence)
#     - the Financials and Ratios tables, parsed as JSON
#   The LLM's URL is validated verbatim against the page HTML (blocks hallucination).
#   Everything else (OCR, chunking, embedding, Pinecone) is unchanged.

import os, re, json, time, hashlib, threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import fitz
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from playwright.sync_api import sync_playwright
from llm import generate_text, LLMUnavailable

load_dotenv()

# ── Clients ───────────────────────────────────────────────────
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
VISION_MODEL  = os.getenv("VISION_MODEL", "gemini-3.1-flash-lite")

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index = pc.Index("psx-financial-intelligence")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# ── Tuning ────────────────────────────────────────────────────
OCR_TEXT_THRESHOLD = 70
OCR_SPARSE_TEXT    = 300
OCR_BATCH_SIZE     = 15
MAX_OCR_WORKERS    = 10
EMBED_BATCH_SIZE   = 78
PINECONE_BATCH     = 200
RENDER_DPI         = 220
CHUNK_SIZE         = 1200
CHUNK_OVERLAP      = 200

PLAYWRIGHT_NAV_TIMEOUT   = 1000
PLAYWRIGHT_CLICK_TIMEOUT = 2000
PLAYWRIGHT_WAIT_TIMEOUT  = 2500
HTTP_TIMEOUT             = 2000

AGENTIC_HTML_MAX_CHARS   = 120_000   # cap on what we send the LLM

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/120.0.0.0 Safari/537.36")

# ══════════════════════════════════════════════════════════════
# Warm browser + background worker (unchanged)
# ══════════════════════════════════════════════════════════════
_PLAYWRIGHT_LOCK = threading.Lock()
_PW = None
_BROWSER = None
_CONTEXT = None

_BG_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ingest")


def _get_context():
    """Return a reusable browser context. Launch only once."""
    global _PW, _BROWSER, _CONTEXT

    if _BROWSER is not None and _BROWSER.is_connected():
        return _CONTEXT

    try:
        if _BROWSER: _BROWSER.close()
    except Exception:
        pass
    try:
        if _PW: _PW.stop()
    except Exception:
        pass

    _PW = sync_playwright().start()
    _BROWSER = _PW.chromium.launch(headless=True, args=[
        "--disable-blink-features=AutomationControlled",
        "--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage",
    ])
    _CONTEXT = _BROWSER.new_context(
        user_agent=UA, viewport={"width": 1440, "height": 900}
    )
    _CONTEXT.route("**/*", lambda route: (
        route.abort() if route.request.resource_type in
        {"image", "font", "media", "stylesheet"}
        else route.continue_()
    ))
    return _CONTEXT


# ── Tiny utilities kept from before ───────────────────────────
_MONTH_RE = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+(\d{1,2}),?\s+(\d{4})",
    re.I,
)
_ISO_RE   = re.compile(r"(\d{4})-(\d{2})-(\d{2})")
_SLASH_RE = re.compile(r"(\d{1,2})[/-](\d{1,2})[/-](\d{4})")


def _parse_any_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    s = str(s).strip()
    m = _ISO_RE.search(s)
    if m:
        try: return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except Exception: pass
    m = _MONTH_RE.search(s)
    if m:
        try:
            month = datetime.strptime(m.group(1)[:3].title(), "%b").month
            return datetime(int(m.group(3)), month, int(m.group(2)))
        except Exception: pass
    m = _SLASH_RE.search(s)
    if m:
        try:
            d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return datetime(y, mo, d)
        except Exception: pass
    return None


def _guess_quarter(dt: datetime) -> str:
    return f"Q{(dt.month - 1) // 3 + 1}"


def _hash_text(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()


def _has_large_image(imgs: List) -> bool:
    for img in imgs:
        try:
            w, h = img[2], img[3]
            if w >= 400 and h >= 400:
                return True
        except (IndexError, TypeError):
            continue
    return False


# ══════════════════════════════════════════════════════════════
# NEW — Level 1 agentic extraction
# ══════════════════════════════════════════════════════════════
AGENTIC_EXTRACT_PROMPT = """You are a PSX (Pakistan Stock Exchange) filings navigator.

You will receive:
  - TICKER
  - TODAY (YYYY-MM-DD)
  - A compact representation of the company page: headings, every <table>,
    every PDF-looking link with its surrounding row text, and (if it fits)
    the cleaned HTML itself.

YOUR JOB
1. Find the LATEST QUARTERLY report PDF for the ticker.
2. Extract the Financials table (if present).
3. Extract the Ratios table (if present).

STRICT RULES FOR THE PDF
- The PDF URL MUST appear verbatim in the input. Never invent, modify,
  or "fix" a URL.
- The URL MUST end in .pdf or contain the word 'download'.
- Prefer quarterly over annual.
- 'Latest' = newest by period_ended date, but ONLY if:
    (a) it is within the last 5 months, OR
    (b) it is from the current year.
  Otherwise return pdf_url = null. Do NOT fall back to a prior-year report.
- Do NOT guess dates. Use only what is in the input.

STRICT RULES FOR TABLES
- Return headers as a list of strings.
- Return rows as a list of lists; each row must match the headers order.
- Skip empty rows, repeated header rows, and disclaimer boilerplate.
- If a table is not present, return null for that field.

OUTPUT — strict JSON only (no markdown, no commentary):
{
  "pdf_url": "<url>" | null,
  "report_type": "Quarterly" | "Annual" | "Unknown" | null,
  "period_ended": "YYYY-MM-DD" | null,
  "posting_date": "YYYY-MM-DD" | null,
  "quarter": "Q1"|"Q2"|"Q3"|"Q4" | null,
  "year": <int> | null,
  "confidence": <float 0.0-1.0>,
  "reason": "<short explanation, <=160 chars>",
  "financials": {"headers": [...], "rows": [[...], ...]} | null,
  "ratios":     {"headers": [...], "rows": [[...], ...]} | null
}
"""


def _prepare_page_for_llm(page) -> str:
    """
    Build a compact but complete view of the page:
      • headings               (navigation context)
      • every <table>          (structured data — the most important part)
      • every PDF-looking link (with row context — the actionable items)
      • cleaned HTML (only if budget remains)
    """
    soup = BeautifulSoup(page.content(), "html.parser")
    for tag in soup(["script", "style", "svg", "noscript", "iframe", "link", "meta"]):
        tag.decompose()

    parts: List[str] = []

    for h in soup.find_all(["h1", "h2", "h3"]):
        txt = h.get_text(" ", strip=True)
        if txt:
            parts.append(f"HEADING: {txt}")

    for i, table in enumerate(soup.find_all("table")):
        table_str = re.sub(r"\s+", " ", str(table))
        parts.append(f"\n=== TABLE {i} ===\n{table_str}")

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not (".pdf" in href.lower() or "download" in href.lower()
                or "/view" in href.lower()):
            continue
        tr = a.find_parent("tr")
        row_text = tr.get_text(" ", strip=True) if tr else ""
        parts.append(f"\n=== PDF LINK ===\nURL: {href}\nROW_TEXT: {row_text[:300]}")

    combined = "\n".join(parts)

    if len(combined) < AGENTIC_HTML_MAX_CHARS * 0.7:
        remaining = AGENTIC_HTML_MAX_CHARS - len(combined) - 100
        full_html = re.sub(r"\s+", " ", str(soup))
        combined += f"\n\n=== PAGE HTML (truncated) ===\n{full_html[:remaining]}"

    return combined[:AGENTIC_HTML_MAX_CHARS]


def _parse_agentic_json(raw: str) -> Optional[dict]:
    if not raw:
        return None
    clean = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            return None


def _validate_table(table) -> Optional[Dict[str, Any]]:
    """Normalize an LLM-returned table into {headers, rows:[{h: v}]}."""
    if not isinstance(table, dict):
        return None
    headers = table.get("headers")
    rows = table.get("rows")
    if not isinstance(headers, list) or not headers:
        return None
    if not isinstance(rows, list) or not rows:
        return None

    headers = [str(h) for h in headers]
    dict_rows = []
    for row in rows:
        if isinstance(row, list):
            d = {}
            for i, h in enumerate(headers):
                val = row[i] if i < len(row) and row[i] is not None else ""
                d[h] = str(val)
            dict_rows.append(d)
        elif isinstance(row, dict):
            d = {}
            for h in headers:
                val = row.get(h)
                d[h] = str(val) if val is not None else ""
            dict_rows.append(d)

    if not dict_rows:
        return None
    return {"headers": headers, "rows": dict_rows}


def _validate_agentic_response(parsed: dict, page_text: str,
                               symbol: str) -> Dict[str, Any]:
    """Validate the LLM output and derive a decision dict."""
    pdf_url = (parsed.get("pdf_url") or "").strip()

    if pdf_url:
        # Hard guard: URL must appear verbatim in the input we sent.
        if pdf_url not in page_text:
            print(f"  ⚠️ LLM returned URL not in page: {pdf_url[:80]}", flush=True)
            pdf_url = ""

        if pdf_url and ".pdf" not in pdf_url.lower() \
                and "download" not in pdf_url.lower():
            print(f"  ⚠️ LLM returned non-PDF URL: {pdf_url[:80]}", flush=True)
            pdf_url = ""

        if pdf_url and not pdf_url.startswith("http"):
            pdf_url = "https://dps.psx.com.pk" + (
                pdf_url if pdf_url.startswith("/") else "/" + pdf_url
            )

    financials = _validate_table(parsed.get("financials"))
    ratios     = _validate_table(parsed.get("ratios"))

    if not pdf_url:
        return {
            "decision": None,
            "financials": financials,
            "ratios": ratios,
            "reason": (parsed.get("reason") or
                       "agent found no qualifying quarterly PDF"),
        }

    period_dt = _parse_any_date(parsed.get("period_ended"))
    quarter   = parsed.get("quarter")
    year      = parsed.get("year")

    if period_dt:
        if quarter not in ("Q1", "Q2", "Q3", "Q4"):
            quarter = _guess_quarter(period_dt)
        if not isinstance(year, int):
            year = period_dt.year

    if quarter not in ("Q1", "Q2", "Q3", "Q4"):
        qm = re.search(r"(Q[1-4])", pdf_url, re.I)
        quarter = qm.group(1).upper() if qm else "Q4"

    if not isinstance(year, int) or year < 2000 or year > 2100:
        ym = re.search(r"(20\d{2})", pdf_url)
        year = int(ym.group(1)) if ym else datetime.today().year

    return {
        "decision": {
            "url": pdf_url,
            "quarter": quarter,
            "year": year,
            "period_ended": (period_dt.strftime("%Y-%m-%d")
                             if period_dt else parsed.get("period_ended")),
            "reason": (parsed.get("reason") or "agentic selection")[:160],
            "confidence": float(parsed.get("confidence") or 0.7),
        },
        "financials": financials,
        "ratios": ratios,
    }


def _agentic_extract(page, symbol: str) -> Optional[Dict[str, Any]]:
    """One LLM call that reads the page and returns {decision, financials, ratios}."""
    try:
        page_text = _prepare_page_for_llm(page)
    except Exception as e:
        print(f"  ⚠️ Could not prepare page for LLM: {e}", flush=True)
        return None

    today = datetime.today().strftime("%Y-%m-%d")
    user_prompt = (
        f"TICKER: {symbol}\n"
        f"TODAY: {today}\n"
        f"CURRENT_YEAR: {datetime.today().year}\n\n"
        f"{page_text}"
    )

    try:
        raw = generate_text(user_prompt, AGENTIC_EXTRACT_PROMPT)
    except LLMUnavailable as e:
        print(f"  ⚠️ Agentic extraction failed (LLM down): {e}", flush=True)
        return None
    except Exception as e:
        print(f"  ⚠️ Agentic extraction failed: {e}", flush=True)
        return None

    parsed = _parse_agentic_json(raw)
    if not parsed:
        print("  ⚠️ Agent returned unparseable JSON", flush=True)
        return None

    return _validate_agentic_response(parsed, page_text, symbol)


# ══════════════════════════════════════════════════════════════
# OCR (unchanged)
# ══════════════════════════════════════════════════════════════
def _ocr_worker_batch(doc, batch_indices: List[int]) -> Dict[int, str]:
    parts = [types.Part(text=(
        f"You are an OCR engine. Extract all text from the "
        f"{len(batch_indices)} page image(s) below.\n"
        'Return strict JSON: {"pages": [{"page": <n>, "text": "..."}, ...]}\n'
        "Rules:\n- Preserve table rows as pipe-separated text.\n"
        "- Do NOT summarize or translate.\n"
        "- 'page' values must match the order of images (1-based).\n"
        "- Extract key metrics but do not fabricate.\n"
    ))]
    for idx in batch_indices:
        pix = doc[idx].get_pixmap(dpi=RENDER_DPI)
        parts.append(types.Part.from_bytes(data=pix.tobytes("png"),
                                            mime_type="image/png"))
    resp = gemini_client.models.generate_content(
        model=VISION_MODEL, contents=parts,
        config=types.GenerateContentConfig(
            temperature=0.2, response_mime_type="application/json"),
    )
    raw = (resp.text or "{}").replace("```json", "").replace("```", "").strip()
    try:
        payload = json.loads(raw or "{}")
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        payload = json.loads(m.group(0)) if m else {}
    out = {}
    for entry in payload.get("pages", []):
        try: n = int(entry.get("page", 0))
        except (ValueError, TypeError): continue
        if 1 <= n <= len(batch_indices):
            out[batch_indices[n - 1]] = entry.get("text", "") or ""
    return out


# ══════════════════════════════════════════════════════════════
# PDF processing (unchanged)
# ══════════════════════════════════════════════════════════════
def _chunk_and_upsert_pdf(pages_text, symbol, quarter, year, source_url=None) -> int:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    collected, seen, chunk_idx = [], set(), 0
    for page_num, text in pages_text.items():
        if not text.strip(): continue
        for chunk in splitter.split_text(text):
            h = _hash_text(chunk)
            if h in seen: continue
            seen.add(h)
            collected.append((page_num, chunk_idx, chunk)); chunk_idx += 1
    if not collected: return 0
    embeddings = embedding_model.encode(
        [c[2] for c in collected], batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True)
    vectors = []
    for (page_num, idx, chunk), emb in zip(collected, embeddings):
        vectors.append({
            "id": f"{symbol}_{year}_{quarter}_p{page_num+1}_chunk_{idx}",
            "values": emb.tolist(),
            "metadata": {
                "text": chunk, "symbol": symbol, "quarter": quarter,
                "year": int(year), "doc_type": "pdf_chunk",
                "source_url": source_url or "Manual Upload",
                "page_number": page_num + 1,
            },
        })
    for i in range(0, len(vectors), PINECONE_BATCH):
        pinecone_index.upsert(vectors=vectors[i:i + PINECONE_BATCH])
    return len(vectors)


def _process_and_ingest_pdf(pdf_bytes, symbol, quarter, year, source_url):
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    n_pages = len(doc)
    page_text, needs_ocr = {}, []
    for i in range(n_pages):
        page = doc[i]
        t = page.get_text("text").strip()
        page_text[i] = t
        imgs = page.get_images(full=True)
        if (len(t) < OCR_TEXT_THRESHOLD
                or (len(t) < OCR_SPARSE_TEXT and _has_large_image(imgs))):
            needs_ocr.append(i)
    if needs_ocr:
        print(f"🔍 OCR: {len(needs_ocr)}/{n_pages} pages...", flush=True)
        batches = [needs_ocr[i:i+OCR_BATCH_SIZE]
                   for i in range(0, len(needs_ocr), OCR_BATCH_SIZE)]
        with ThreadPoolExecutor(max_workers=MAX_OCR_WORKERS) as pool:
            futures = {pool.submit(_ocr_worker_batch, doc, b): b for b in batches}
            for fut in as_completed(futures):
                try:
                    for idx, text in fut.result().items():
                        if text.strip():
                            native = page_text.get(idx, "").strip()
                            page_text[idx] = (text if not native
                                              else native + "\n\n[Image Extracted]\n" + text)
                except Exception as e:
                    print(f"⚠️ OCR batch failed: {e}", flush=True)
    doc.close()
    if sum(len(t) for t in page_text.values()) == 0:
        raise ValueError("No extractable text found in PDF.")
    unique, seen = {}, set()
    for page_num, text in page_text.items():
        if not text.strip(): continue
        h = _hash_text(text)
        if h in seen: continue
        seen.add(h); unique[page_num] = text
    chunks_count = _chunk_and_upsert_pdf(unique, symbol, quarter, year, source_url)
    return {"symbol": symbol, "quarter": quarter, "year": year,
            "pages": n_pages, "ocr_pages": len(needs_ocr),
            "chunks": chunks_count, "source_url": source_url}


# ══════════════════════════════════════════════════════════════
# Metrics (unchanged)
# ══════════════════════════════════════════════════════════════
def _financials_to_text(symbol, financials, as_of):
    if not financials or not financials.get("rows"): return None
    lines = [f"{symbol} FINANCIAL METRICS (source: PSX, as of {as_of}):"]
    lines.append("  " + " | ".join(financials["headers"]))
    for row in financials["rows"]:
        lines.append("  " + " | ".join(row.get(h, "") for h in financials["headers"]))
    return "\n".join(lines)


def _ratios_to_text(symbol, ratios, as_of):
    if not ratios or not ratios.get("rows"): return None
    lines = [f"{symbol} FINANCIAL RATIOS (source: PSX, as of {as_of}):"]
    lines.append("  " + " | ".join(ratios["headers"]))
    for row in ratios["rows"]:
        lines.append("  " + " | ".join(row.get(h, "") for h in ratios["headers"]))
    return "\n".join(lines)


def _upsert_metrics(symbol, docs, source_url):
    if not docs: return 0
    embeddings = embedding_model.encode(
        [d[1] for d in docs], batch_size=EMBED_BATCH_SIZE,
        show_progress_bar=False, convert_to_numpy=True)
    vectors = []
    for (doc_type, text), emb in zip(docs, embeddings):
        vectors.append({
            "id": f"{symbol}_metrics_{doc_type}",
            "values": emb.tolist(),
            "metadata": {"text": text, "symbol": symbol, "doc_type": doc_type,
                         "source_url": source_url,
                         "as_of": datetime.now().strftime("%Y-%m-%d")},
        })
    for i in range(0, len(vectors), PINECONE_BATCH):
        pinecone_index.upsert(vectors=vectors[i:i + PINECONE_BATCH])
    return len(vectors)


# ══════════════════════════════════════════════════════════════
# Background ingestion (unchanged)
# ══════════════════════════════════════════════════════════════
def _async_ingest(pdf_bytes, symbol, decision, financials, ratios, home_url):
    try:
        if pdf_bytes and decision:
            _process_and_ingest_pdf(pdf_bytes, symbol,
                                     decision["quarter"], decision["year"],
                                     decision["url"])
        as_of = datetime.now().strftime("%Y-%m-%d")
        docs = []
        t = _financials_to_text(symbol, financials, as_of)
        if t: docs.append(("financial_metrics", t))
        t = _ratios_to_text(symbol, ratios, as_of)
        if t: docs.append(("ratios", t))
        _upsert_metrics(symbol, docs, home_url)
        print(f"  ✅ [bg] {symbol} ingestion finished", flush=True)
    except Exception as e:
        print(f"  ⚠️ [bg] {symbol} ingest failed: {e}", flush=True)


# ══════════════════════════════════════════════════════════════
# PUBLIC ENTRY
# ══════════════════════════════════════════════════════════════
def download_report_agent(symbol: str) -> Dict[str, Any]:
    symbol = symbol.upper().strip()
    t0 = time.time()
    print(f"\n🤖 download_report_agent({symbol})", flush=True)

    home_url = f"https://dps.psx.com.pk/company/{symbol}"
    pdf_bytes = None
    decision = None
    financials = ratios = None

    with _PLAYWRIGHT_LOCK:
        try:
            context = _get_context()
            page = context.new_page()
        except Exception as e:
            print(f"  ⚠️ Browser launch failed: {e} — retrying", flush=True)
            global _BROWSER, _CONTEXT
            _BROWSER = _CONTEXT = None
            context = _get_context()
            page = context.new_page()

        try:
            # ---- Navigate ----
            print(f"  [1/3] Opening {home_url}", flush=True)
            page.goto(home_url, wait_until="commit",
                      timeout=PLAYWRIGHT_NAV_TIMEOUT)

            # Wait until a table with the right signature is present.
            try:
                page.wait_for_function("""
                    () => {
                      const tables = document.querySelectorAll('table');
                      for (const t of tables) {
                        const txt = (t.innerText || '').toLowerCase();
                        if (txt.includes('period ended') || txt.includes('posting'))
                          return true;
                      }
                      return false;
                    }
                """, timeout=PLAYWRIGHT_WAIT_TIMEOUT)
            except Exception:
                pass

            if len(page.content()) < 500:
                raise RuntimeError(f"PSX blank page for {symbol}")

            # ---- Try to open the REPORTS tab ----
            for sel in ["a:has-text('REPORTS')", "a:has-text('Reports')",
                        "text=Reports", "a[href*='#reports']",
                        "li:has-text('Reports') a"]:
                try:
                    page.click(sel, timeout=PLAYWRIGHT_CLICK_TIMEOUT)
                    try:
                        page.wait_for_selector("table",
                                                timeout=PLAYWRIGHT_WAIT_TIMEOUT)
                    except Exception:
                        pass
                    break
                except Exception:
                    continue

            # ---- Agentic extraction (ONE LLM call) ----
            print(f"  [2/3] Agentic extraction (reading page)...", flush=True)
            t_ext = time.time()
            extracted = _agentic_extract(page, symbol)
            if extracted:
                decision   = extracted.get("decision")
                financials = extracted.get("financials")
                ratios     = extracted.get("ratios")
            else:
                decision = financials = ratios = None

            print(
                f"        agent → "
                f"{'PDF: ' + decision['url'][:80] if decision else 'no PDF'} · "
                f"financials: {'y' if financials else 'n'} · "
                f"ratios: {'y' if ratios else 'n'} "
                f"({time.time() - t_ext:.1f}s)",
                flush=True,
            )
            if decision:
                print(f"        reason: {decision.get('reason', '')}", flush=True)

            # ---- Download ----
            if decision:
                try:
                    resp = context.request.get(decision["url"],
                                                timeout=HTTP_TIMEOUT)
                    if resp.ok and resp.body()[:4] == b"%PDF":
                        pdf_bytes = resp.body()
                except Exception as e:
                    print(f"  ⚠️ in-browser download failed: {e}", flush=True)
                if not pdf_bytes:
                    try:
                        r = requests.get(decision["url"],
                                          headers={"User-Agent": UA},
                                          timeout=HTTP_TIMEOUT)
                        if r.ok and r.content[:4] == b"%PDF":
                            pdf_bytes = r.content
                    except Exception as e:
                        print(f"  ⚠️ requests download failed: {e}", flush=True)
        finally:
            try: page.close()
            except Exception: pass

    # ---- Return fast: queue heavy work in background ----
    result = {
        "symbol": symbol,
        "pdf_queued": False,
        "pdf_ingested": False,
        "metrics_ingested": 0,
        "reason": (decision.get("reason") if decision
                   else "agent found no qualifying quarterly PDF"),
        "source_url": decision.get("url") if decision else None,
        "quarter": decision.get("quarter") if decision else None,
        "year": decision.get("year") if decision else None,
        "confidence": decision.get("confidence") if decision else None,
    }

    if pdf_bytes and decision:
        _BG_POOL.submit(_async_ingest, pdf_bytes, symbol, decision,
                         financials, ratios, home_url)
        result["pdf_queued"] = True
    else:
        _BG_POOL.submit(_async_ingest, None, symbol, None,
                         financials, ratios, home_url)

    print(f"  ✅ Returned in {time.time()-t0:.1f}s "
          f"(ingestion running in background)", flush=True)
    return result


# ── Backwards-compat shims (unchanged) ────────────────────────
def find_quarterly_reports(symbol: str, limit: int = 8) -> List[Dict]:
    result = download_report_agent(symbol.upper().strip())
    if result.get("pdf_queued") and result.get("source_url"):
        return [{"url": result["source_url"],
                 "quarter": result.get("quarter"),
                 "year": result.get("year")}]
    return []


def auto_ingest_psx_report(symbol: str) -> Dict[str, Any]:
    return download_report_agent(symbol)


def ingest_quarterly_pdf(file_path: str, symbol: str, quarter: str, year: int) -> None:
    with open(file_path, "rb") as f:
        pdf_bytes = f.read()
    _process_and_ingest_pdf(pdf_bytes, symbol.upper().strip(), quarter, year,
                             "Manual Upload")


# ── CLI (unchanged) ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    symbol = sys.argv[1] if len(sys.argv) > 1 else "PSO"

    print("🔥 Warming browser...", flush=True)
    try:
        with _PLAYWRIGHT_LOCK:
            ctx = _get_context()
            pg = ctx.new_page()
            pg.goto("https://dps.psx.com.pk/company/PSO",
                    wait_until="commit", timeout=PLAYWRIGHT_NAV_TIMEOUT)
            pg.close()
        print("🔥 Browser ready.", flush=True)
    except Exception as e:
        print(f"🔥 Warmup skipped: {e}", flush=True)

    t0 = time.time()
    try:
        result = download_report_agent(symbol)
        print(f"\n✅ Returned in {time.time()-t0:.1f}s")
        print(json.dumps(result, indent=2, default=str))
        print("\n⏳ Waiting 15s for background ingestion to finish (debug only)...")
        time.sleep(15)
    except Exception as e:
        print(f"\n❌ Failed in {time.time()-t0:.1f}s: {e}")