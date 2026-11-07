# ingestion.py - News ETL Pipeline for PostgreSQL
"""
News ingestion — RSS + HTML scraping, deduped by URL hash, safe date handling.

Production scheduling (pick ONE — see bottom of file for setup):
  1. APScheduler (in-process, easiest, best for single-worker)
  2. systemd timer (recommended — exact 10h intervals, survives restarts)
  3. cron (fallback — approximate 10h with 4h gap at midnight)

Manual one-shot:  python ingestion.py
In-process mode:  python ingestion.py --schedule
"""

import os
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin

import feedparser
import requests
from bs4 import BeautifulSoup
from sqlalchemy import text

# ✅ Unified DB module — one engine for the whole app
from db import engine


# ============================================================
# SOURCES
# ============================================================
SOURCES = {
    "Business Recorder": {
        "url": "https://www.brecorder.com/feeds/latest-news/",
        "type": "rss",
    },
    "Investify News": {
        "url": "https://www.investify.pk/news",
        "type": "html",
        "base_url": "https://www.investify.pk",
    },
    "PSX Terminal": {
        "url": "https://psxterminal.com/news",
        "type": "html",
        "base_url": "https://psxterminal.com",
    },
    "Business Recorder Trends": {
        "url": "https://www.brecorder.com/trends/pakistan-stock-exchange",
        "type": "html",
        "base_url": "https://www.brecorder.com",
    },
    "Dawn Business": {
        "url": "https://www.dawn.com/newspaper/business-finance",
        "type": "rss",
        "base_url": "https://www.dawn.com",
    },
}


# URLs we do NOT want to ingest (navigation, social, etc.)
_SKIP_URL_PARTS = {
    "/about", "/contact", "/privacy", "/terms", "/login", "/signup",
    "/subscribe", "/advertise", "/careers", "/faq", "/help", "/search",
    "facebook.com", "twitter.com", "linkedin.com", "instagram.com",
    "youtube.com", "whatsapp.com", "mailto:", "javascript:",
}

MAX_ITEMS_PER_SOURCE = 20
HTTP_TIMEOUT_S = 15
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


# ============================================================
# HELPERS
# ============================================================
def generate_url_hash(url: str) -> str:
    """SHA-256 hash of URL for deduplication."""
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def _rss_date_to_iso(entry) -> Optional[str]:
    """
    Convert feedparser's published_parsed struct to an ISO 8601 string.
    Falls back to updated_parsed, then None. NEVER uses 'now'.
    """
    struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if not struct:
        return None
    try:
        return datetime(*struct[:6], tzinfo=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return None


def _html_find_date(anchor) -> Optional[str]:
    """
    Look for a <time datetime="..."> element near the anchor (parent or grandparent).
    Many news sites embed schema.org datetime. Returns ISO string or None.
    NEVER uses 'now'.
    """
    node = anchor.parent
    for _ in range(3):  # walk up to 3 levels
        if node is None:
            break
        time_el = node.find("time") if hasattr(node, "find") else None
        if time_el:
            dt = time_el.get("datetime") or time_el.get_text(strip=True)
            if dt:
                return dt
        node = node.parent if hasattr(node, "parent") else None
    return None


# ============================================================
# SOURCE PARSERS
# ============================================================
def _parse_rss(source_name: str, config: dict) -> list[dict]:
    feed = feedparser.parse(config["url"])
    out = []
    for entry in feed.entries:
        title = entry.get("title", "").strip()
        link = entry.get("link", "").strip()
        if not title or not link:
            continue
        out.append({
            "title": title,
            "link": link,
            "summary": entry.get("summary", ""),
            "published": _rss_date_to_iso(entry),  # None if unknown
        })
    return out


def _parse_html(source_name: str, config: dict) -> list[dict]:
    try:
        resp = requests.get(
            config["url"],
            headers={"User-Agent": USER_AGENT},
            timeout=HTTP_TIMEOUT_S,
        )
        if resp.status_code != 200:
            print(f"   ⚠️ HTTP {resp.status_code} from {source_name}")
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
    except Exception as e:
        print(f"   ⚠️ Fetch failed for {source_name}: {e}")
        return []

    base_url = config.get("base_url", config["url"])
    out = []
    for anchor in soup.find_all("a", href=True):
        link = anchor["href"]
        title = anchor.get_text(" ", strip=True)

        if len(title) < 20:
            continue
        if any(bad in link.lower() for bad in _SKIP_URL_PARTS):
            continue
        if not link.startswith("http"):
            link = urljoin(base_url, link)

        out.append({
            "title": title,
            "link": link,
            "summary": title,
            # Try real date from <time>; else None (never fake "now")
            "published": _html_find_date(anchor),
        })

        if len(out) >= MAX_ITEMS_PER_SOURCE:
            break
    return out


# ============================================================
# PERSISTENCE — per-row transaction so one bad article doesn't kill the batch
# ============================================================
def _insert_article(article: dict, source_name: str) -> bool:
    """Insert one article. Returns True if newly inserted, False if duplicate/error."""
    title = article["title"]
    link = article["link"]
    summary = article["summary"]
    published = article["published"]  # ISO string or None

    if not link:
        return False

    url_hash = generate_url_hash(link)
    clean_text = BeautifulSoup(summary or "", "html.parser").get_text()

    try:
        with engine.begin() as conn:
            result = conn.execute(
                text("""
                    INSERT INTO news_articles
                        (headline, full_text, source, url, url_hash, published_date, sentiment)
                    VALUES
                        (:title, :full_text, :source, :url, :url_hash, :published_date, 'Neutral')
                    ON CONFLICT (url_hash) DO NOTHING
                """),
                {
                    "title": title[:500],
                    "full_text": clean_text[:50000],
                    "source": source_name,
                    "url": link,
                    "url_hash": url_hash,
                    "published_date": published,  # ISO string OR None — both valid
                },
            )
            return result.rowcount > 0
    except Exception as e:
        print(f"   ⚠️ Insert failed for {link[:80]}: {e}")
        return False


# ============================================================
# MAIN ETL RUN
# ============================================================
def ingest_rss_news() -> dict:
    """
    Fetch every source, parse, dedupe, persist.
    Returns a summary dict: {source: {"fetched": N, "inserted": M, "error": str|None}}
    """
    started = datetime.now(timezone.utc)
    print(f"\n{'='*60}")
    print(f"📰 News ingestion started at {started.isoformat()}")
    print(f"{'='*60}")

    summary: dict[str, dict] = {}
    total_inserted = 0

    for source_name, config in SOURCES.items():
        print(f"\n📥 {source_name}...")
        source_result = {"fetched": 0, "inserted": 0, "error": None}

        try:
            if config["type"] == "rss":
                articles = _parse_rss(source_name, config)
            elif config["type"] == "html":
                articles = _parse_html(source_name, config)
            else:
                source_result["error"] = f"unknown type: {config['type']}"
                summary[source_name] = source_result
                continue

            source_result["fetched"] = len(articles)

            for art in articles:
                if _insert_article(art, source_name):
                    source_result["inserted"] += 1

            total_inserted += source_result["inserted"]
            print(f"   ✓ fetched={source_result['fetched']} · inserted={source_result['inserted']}")

        except Exception as e:
            source_result["error"] = str(e)[:200]
            print(f"   ❌ {source_name} failed: {e}")

        summary[source_name] = source_result

    elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    print(f"\n{'='*60}")
    print(f"✅ Total new articles: {total_inserted} · elapsed: {elapsed:.1f}s")
    print(f"{'='*60}\n")

    return summary


# ============================================================
# PRODUCTION SCHEDULER — pick ONE option below
# ============================================================
def _run_scheduler():
    """
    OPTION A — In-process scheduler (APScheduler).

    Requirements:
        pip install apscheduler

    Behavior:
        - Runs immediately on start, then every 10 hours
        - Runs as a blocking process (dedicated worker, not the web server)
        - Safe: single process, no overlap

    Caveat:
        If the process dies, no scrape happens until it's restarted.
        If you run multiple copies, you get multiple scrapes per cycle.

    Launch:
        python ingestion.py --schedule
        # or under systemd/supervisor
    """
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        print("❌ APScheduler not installed.")
        print("   pip install apscheduler")
        raise

    scheduler = BlockingScheduler(timezone="UTC")

    # next_run_time=datetime.now() → run once immediately at startup
    scheduler.add_job(
        ingest_rss_news,
        trigger=IntervalTrigger(hours=10),
        next_run_time=datetime.now(),
        id="news_ingest",
        name="PSX news ingestion",
        max_instances=1,          # never overlap
        coalesce=True,            # if missed, run once (not 3 times)
        misfire_grace_time=600,   # allow 10-min late start
    )

    print("🕐 Scheduler started — news ingestion every 10 hours (UTC)")
    print("   Press Ctrl+C to stop.\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n👋 Scheduler stopped.")
        scheduler.shutdown(wait=False)


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="PSX news ingestion")
    parser.add_argument(
        "--schedule", action="store_true",
        help="Run continuously, scraping every 10 hours (requires apscheduler)",
    )
    args = parser.parse_args()

    if args.schedule:
        _run_scheduler()
    else:
        ingest_rss_news()


if __name__ == "__main__":
    main()