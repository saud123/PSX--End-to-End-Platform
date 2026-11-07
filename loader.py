# loader.py - Market Price Loader for PostgreSQL
from datetime import datetime, timedelta
import pandas as pd
import psxdata
from sqlalchemy import create_engine, text
import warnings
import os

# Suppress psxdata warnings so they don't clutter the terminal
warnings.filterwarnings("ignore")

# PostgreSQL Database Connection URL
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_URL = f"postgresql+psycopg2://postgres:{DB_PASSWORD}@localhost:5432/psx_intelligence"



def load_psx_market_prices(target_symbol: str = None):
    engine = create_engine(DB_URL)
    current_date = datetime.today().date()
    
    default_symbols = ["ENGRO", "OGDC", "LUCK", "KEL"]
    symbols_to_scrape = [target_symbol.strip().upper()] if target_symbol else default_symbols

    for symbol in symbols_to_scrape:
        print(f"📈 Fetching historical market data for {symbol} via psxdata...")
        try:
            # 1. Find the last date we have in the DB
            with engine.connect() as conn:
                res = conn.execute(
                    text("SELECT MAX(trade_date) FROM market_prices_eod WHERE symbol = :s"), 
                    {"s": symbol}
                )
                last_date_row = res.fetchone()[0]
            
            # 2. Determine start date (avoid requesting future dates)
            if last_date_row:
                if last_date_row >= current_date:
                    print(f"✅ {symbol} is already up to date (Last date: {last_date_row}).")
                    continue
                start_date = (last_date_row + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                start_date = "2024-01-01"

            # 3. Fetch the data
            df = psxdata.stocks(symbol, start=start_date, end=current_date.strftime('%Y-%m-%d'))
            
            if df is None or df.empty:
                print(f"✅ No new price data found for {symbol}.")
                continue

            # 4. Clean the data (Fixes the OHLC constraint & Chronological order issues)
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date']) # Drop rows with bad dates
                df = df.sort_values('date')     # Sort by date
            else:
                df = df.reset_index()
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df = df.dropna(subset=['date'])
                df = df.sort_values('date')

            # Ensure numeric columns are actually numbers
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

            total_loaded = 0
            with engine.connect() as conn:
                for _, row in df.iterrows():
                    date_val = row.get('date')
                    date_str = date_val.strftime("%Y-%m-%d") if hasattr(date_val, 'strftime') else str(date_val).split("T")[0]

                    # Skip if we somehow grabbed a future date
                    if date_str > current_date.strftime('%Y-%m-%d'):
                        continue

                    open_val = float(row.get('open', 0))
                    high_val = float(row.get('high', 0))
                    low_val = float(row.get('low', 0))
                    close_val = float(row.get('close', 0))
                    volume_val = int(row.get('volume', 0)) if 'volume' in row else 0

                    result = conn.execute(text("""
                        INSERT INTO market_prices_eod (symbol, trade_date, open, high, low, close, volume)
                        VALUES (:symbol, :trade_date, :open, :high, :low, :close, :volume)
                        ON CONFLICT (symbol, trade_date) 
                        DO UPDATE SET 
                            open = EXCLUDED.open,
                            high = EXCLUDED.high,
                            low = EXCLUDED.low,
                            close = EXCLUDED.close,
                            volume = EXCLUDED.volume
                    """), {
                        "symbol": symbol,
                        "trade_date": date_str,
                        "open": open_val,
                        "high": high_val,
                        "low": low_val,
                        "close": close_val,
                        "volume": volume_val
                    })
                    
                    if result.rowcount > 0:
                        total_loaded += 1

                conn.commit()

            print(f"✅ Successfully processed {total_loaded} new records for {symbol}")

        except Exception as e:
            print(f"❌ Error loading market data for {symbol}: {e}")

    print(f"🎉 Finished loading/updating price records in PostgreSQL!")

if __name__ == "__main__":
    load_psx_market_prices()