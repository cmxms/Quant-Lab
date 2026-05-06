import os
import pandas as pd
import yfinance as yf
from datetime import datetime
import shutil

# Configuration
PARQUET_FILENAME = 'glbx-mdp3-20210505-20260505.ohlcv-1m.parquet'
PARQUET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), PARQUET_FILENAME)
SYMBOL = 'NQ=F'

def update_data():
    if not os.path.exists(PARQUET_PATH):
        print(f"Error: Parquet file not found at {PARQUET_PATH}")
        return

    # 1. Load existing data and get last timestamp
    print("Loading master Parquet file...")
    try:
        df_master = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"Error reading Parquet: {e}")
        return

    if df_master.empty:
        print("Parquet file appears to be empty.")
        return
        
    last_ts_str = df_master['ts_event'].iloc[-1]
    
    # Parse last timestamp and ensure UTC
    try:
        last_ts = pd.to_datetime(last_ts_str)
        if last_ts.tz is None:
            last_ts = last_ts.tz_localize('UTC')
        else:
            last_ts = last_ts.tz_convert('UTC')
        print(f"Last timestamp in master file: {last_ts}")
    except Exception as e:
        print(f"Error parsing last timestamp: {e}")
        return

    # 2. Download new data from yfinance
    print(f"Downloading fresh 1m data for {SYMBOL} from Yahoo Finance...")
    # interval='1m' allows up to 7 days
    df_new = yf.download(SYMBOL, interval='1m', period='5d', progress=False)
    
    if df_new.empty:
        print("No data received from Yahoo Finance.")
        return

    # 3. Clean and filter
    # Ensure index is UTC
    if getattr(df_new.index, 'tz', None) is None:
        df_new.index = df_new.index.tz_localize('UTC')
    else:
        df_new.index = df_new.index.tz_convert('UTC')

    # Keep only data strictly newer than last_ts
    df_new = df_new[df_new.index > last_ts]

    if df_new.empty:
        print("Data is already up to date. No new rows to append.")
        return

    print(f"Found {len(df_new)} new 1-minute bars.")

    # 4. Map to original schema
    def get_col(df, name):
        col_data = df[name]
        if isinstance(col_data, pd.DataFrame):
            return col_data.iloc[:, 0]
        return col_data

    update_data = {
        'ts_event': df_new.index.strftime('%Y-%m-%dT%H:%M:%S.000000000Z'),
        'rtype': 33,
        'publisher_id': 1,
        'instrument_id': 0, # Placeholder for Databento specific ID
        'open': get_col(df_new, 'Open'),
        'high': get_col(df_new, 'High'),
        'low': get_col(df_new, 'Low'),
        'close': get_col(df_new, 'Close'),
        'volume': get_col(df_new, 'Volume').astype(int),
        'symbol': SYMBOL
    }
    
    update_df = pd.DataFrame(update_data)

    # Combine master and update
    df_combined = pd.concat([df_master, update_df], ignore_index=True)

    # 5. SAFE SAVE LOGIC: Protect the master file!
    # Write to temp file first. If successful, back up the old master, then rename.
    temp_path = PARQUET_PATH + ".tmp"
    backup_path = PARQUET_PATH + ".bak"
    
    max_retries = 5
    retry_delay = 5 # seconds
    
    for attempt in range(max_retries):
        try:
            # 5a. Save to temp
            df_combined.to_parquet(temp_path, index=False)
            
            # 5b. Backup current master
            if os.path.exists(backup_path):
                os.remove(backup_path)
            shutil.copy2(PARQUET_PATH, backup_path)
            
            # 5c. Replace master with temp
            os.replace(temp_path, PARQUET_PATH)
            
            print(f"Successfully updated {PARQUET_PATH}")
            print(f"Appended records from {update_df['ts_event'].iloc[0]} to {update_df['ts_event'].iloc[-1]}")
            
            # Warning about data quality mismatch
            print("\n" + "!"*50)
            print("WARNING: Data Quality Mismatch Note")
            print("The existing data is Databento (CME Direct).")
            print("The appended data is Yahoo Finance (Aggregated).")
            print("Expect minor discrepancies in price and volume profiles.")
            print("!"*50)
            break
        except PermissionError:
            print(f"Attempt {attempt + 1}: File is locked (likely by OneDrive). Retrying in {retry_delay}s...")
            import time
            time.sleep(retry_delay)
        except Exception as e:
            print(f"Failed to write safely: {e}")
            # Try to clean up temp file if something broke
            if os.path.exists(temp_path):
                os.remove(temp_path)
            break
    else:
        print(f"Error: Could not finalize save after {max_retries} attempts.")

if __name__ == "__main__":
    update_data()
