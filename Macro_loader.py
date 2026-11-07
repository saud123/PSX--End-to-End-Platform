import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright
import time

load_dotenv()

DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL = f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/psx_intelligence"
engine = create_engine(DB_URL)

TARGET_URLS = [
    "https://economy.finance.gov.pk/growth-and-investment",
    "https://economy.finance.gov.pk/agriculture",
    "https://economy.finance.gov.pk/",
    "https://economy.finance.gov.pk/economic-and-social-indicators",
    "https://economy.finance.gov.pk/money-and-credit",
    "https://economy.finance.gov.pk/manufacturing-and-mining",
    "https://economy.finance.gov.pk/capital-markets-and-corporate-sector",
    "https://economy.finance.gov.pk/fiscal-development",
    "https://economy.finance.gov.pk/inflation",
    "https://economy.finance.gov.pk/trade-and-payments",
    "https://economy.finance.gov.pk/public-debt",
    "https://economy.finance.gov.pk/education",
    "https://economy.finance.gov.pk/energy",
    "https://economy.finance.gov.pk/health-and-nutrition",
    "https://economy.finance.gov.pk/population-labor-force-and-employment",
    "https://economy.finance.gov.pk/transport-and-communications",
    "https://economy.finance.gov.pk/energy",
    "https://economy.finance.gov.pk/information-technology-and-telecommunication"
]

all_dataframes = []





def init_db_table():
    with engine.connect() as conn:
        # 1. Create the main table safely (no DROP)
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS finance_macro_dashboard (
                id SERIAL PRIMARY KEY,
                source_url TEXT,
                page_category TEXT,
                chart_title TEXT,
                raw_data JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))
        
        # 2. Add a UNIQUE constraint so we can UPSERT later
        # (This ensures we only have ONE row per chart title)
        try:
            conn.execute(text("""
                ALTER TABLE finance_macro_dashboard 
                ADD CONSTRAINT unique_chart UNIQUE (page_category, chart_title);
            """))
        except Exception:
            pass # Constraint already exists
        
        conn.commit()






# ####
def scrape_finance_dashboard():
    print("🇵🇰 Starting Ministry of Finance Dashboard Scraper via Playwright...")
    init_db_table()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False) # Set to False so you can watch it work!
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        for url in TARGET_URLS:
            page_name = url.split('/')[-1]
            print(f"\n📥 Scraping: {page_name}...")
            
            try:
                page.goto(url, wait_until="networkidle", timeout=60000)
                
                # Scroll down to trigger lazy loading of charts
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                time.sleep(3) # Wait for charts to render
                
                # Find all ApexCharts toolbars on the page
                charts = page.query_selector_all('.apexcharts-canvas')
                
                if not charts:
                    print(f"   ⚠️ No charts found on {page_name}.")
                    continue
                
                print(f"   ✅ Found {len(charts)} charts on {page_name}.")
                
                for i, chart in enumerate(charts):
                    try:
                        # Hover over the chart to reveal the toolbar menu
                        chart.hover()
                        time.sleep(0.5)
                        
                        # Find the chart title (usually in a <p> tag just above the chart)
                        parent = chart.evaluate_handle("node => node.parentElement")
                        title_el = parent.as_element().query_selector('p')
                        chart_title = title_el.inner_text() if title_el else f"Chart_{i}"
                        
                        # Click the ApexCharts menu icon
                        menu_icon = chart.query_selector('.apexcharts-menu-icon')
                        if menu_icon:
                            menu_icon.click()
                            time.sleep(0.5)
                            
                            # Click the "Download CSV" button
                            with page.expect_download() as download_info:
                                chart.query_selector('.exportCSV').click()
                            
                            download = download_info.value
                            csv_path = f"temp_{page_name}_{i}.csv"
                            download.save_as(csv_path)
                            
                            # Read the downloaded CSV
                            df = pd.read_csv(csv_path)
                            
                            # Add metadata
                            df['source_url'] = url
                            df['page_category'] = page_name
                            df['chart_title'] = chart_title
                            
                            all_dataframes.append(df)
                            os.remove(csv_path) # Clean up temp file
                            
                            print(f"      ✅ Downloaded: {chart_title}")
                            
                    except Exception as e:
                        print(f"      ⚠️ Could not extract chart {i}: {e}")
                        
            except Exception as e:
                print(f"   ❌ Failed to load page {page_name}: {e}")
                
        browser.close()

    # 3. Save to PostgreSQL (Using JSONB for unstructured data)

    if all_dataframes:
        print("\n🔄 Combining all scraped charts...")
        final_df = pd.concat(all_dataframes, ignore_index=True)
        
        print("🚀 Pushing to PostgreSQL database...")
        try:
            metadata_cols = ['source_url', 'page_category', 'chart_title']
            data_cols = [c for c in final_df.columns if c not in metadata_cols]
            
            final_df['raw_data'] = final_df[data_cols].apply(lambda row: row.to_json(), axis=1)
            final_df = final_df[metadata_cols + ['raw_data']]
            
            # ✅ PRODUCTION FIX: Use raw SQL to UPSERT instead of append
            with engine.begin() as conn:
                for _, row in final_df.iterrows():
                    conn.execute(text("""
                        INSERT INTO finance_macro_dashboard (source_url, page_category, chart_title, raw_data)
                        VALUES (:url, :cat, :title, :data)
                        ON CONFLICT (page_category, chart_title) 
                        DO UPDATE SET 
                            raw_data = EXCLUDED.raw_data,
                            created_at = CURRENT_TIMESTAMP;
                    """), {
                        "url": row['source_url'],
                        "cat": row['page_category'],
                        "title": row['chart_title'],
                        "data": row['raw_data']
                    })
                    
            print(f"✅ Successfully UPSERTED {len(final_df)} chart datasets to PostgreSQL!")
        except Exception as e:
            print(f"❌ Database insertion failed: {e}")

if __name__ == "__main__":
    scrape_finance_dashboard()