import argparse
import os
import sys
import pandas as pd
import pandas_ta as ta

def main():
    parser = argparse.ArgumentParser(description="Generate stationary machine learning features from OHLCV 1-minute data in parquet format.")
    parser.add_argument("--input", required=True, help="Path to input Parquet file")
    parser.add_argument("--output", required=True, help="Path to output Parquet file")
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    print(f"Loading data from {args.input}...")
    try:
        df = pd.read_parquet(args.input)
    except Exception as e:
        print(f"Error reading Parquet: {e}")
        sys.exit(1)
        
    # Identify datetime column and set as index
    datetime_cols = [c for c in df.columns if c.lower() in ['ts_event', 'ts', 'date', 'datetime', 'timestamp', 'time']]
    if not isinstance(df.index, pd.DatetimeIndex):
        if datetime_cols:
            print(f"Using '{datetime_cols[0]}' as datetime column.")
            df[datetime_cols[0]] = pd.to_datetime(df[datetime_cols[0]], utc=True)
            df.set_index(datetime_cols[0], inplace=True)
        else:
            try:
                df.index = pd.to_datetime(df.index, utc=True)
            except Exception:
                print("Error: Could not identify a datetime index. Ensure the Parquet file has a 'datetime' or 'date' column.")
                sys.exit(1)
                
    # Handle multiple symbols if present
    if 'symbol' in df.columns:
        unique_symbols = df['symbol'].unique()
        if len(unique_symbols) > 1:
            print(f"Warning: Multiple symbols found: {unique_symbols}")
            # If this is continuous stitched data (it might have multiple symbols preserved), 
            # we should NOT filter if the user passed it explicitly as continuous.
            # We will just warn and proceed without filtering.
            print("Proceeding without filtering symbols (assuming this is stitched continuous data).")
                
    # Timezone handling: Ensure US/Eastern
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
        
    if getattr(df.index, 'tz', None) is None:
        try:
            df.index = df.index.tz_localize('US/Eastern')
        except Exception:
            df.index = df.index.tz_localize('US/Eastern', ambiguous='NaT', nonexistent='shift_forward')
    else:
        df.index = df.index.tz_convert('US/Eastern')
        
    # Check for required columns
    required_columns = {'open', 'high', 'low', 'close', 'volume'}
    col_map = {c.lower(): c for c in df.columns}
    
    missing_cols = required_columns - set(col_map.keys())
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
        
    # Ensure standard names for OHLCV
    df.rename(columns={
        col_map['open']: 'Open',
        col_map['high']: 'High',
        col_map['low']: 'Low',
        col_map['close']: 'Close',
        col_map['volume']: 'Volume'
    }, inplace=True)
    
    # Sort index chronologically to ensure accurate rolling calculations
    df.sort_index(inplace=True)
    
    print("Calculating features...")
    
    # --- Base Indicators using pandas_ta ---
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=200, append=True)
    
    # VWAP (pandas_ta groups by day by default using the DatetimeIndex)
    df.ta.vwap(append=True) 
    
    df.ta.rsi(length=14, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    
    # 390-period SMA for Volume (representing ~1 full trading day of 1m data)
    df['SMA_Volume_390'] = df.ta.sma(close=df['Volume'], length=390)
    
    df.ta.adx(length=14, append=True)
    
    # --- Exact Feature Transformations ---
    
    # 1-3. Trend Features
    df['9EMA_Dist'] = (df['Close'] - df['EMA_9']) / df['Close']
    df['200EMA_Dist'] = (df['Close'] - df['EMA_200']) / df['Close']
    df['EMA_Spread'] = (df['EMA_9'] - df['EMA_20']) / df['EMA_20']
    
    # 4. VWAP_Dist
    vwap_col = [c for c in df.columns if 'VWAP' in c][0]
    df['VWAP_Dist'] = (df['Close'] - df[vwap_col]) / df[vwap_col]
    
    # 5-8. Momentum & Exhaustion Features
    rsi_col = [c for c in df.columns if 'RSI' in c][0]
    df['RSI_14'] = df[rsi_col]
    
    willr_col = [c for c in df.columns if 'WILLR' in c][0]
    df['WillR_14'] = df[willr_col]
    
    macd_hist_col = [c for c in df.columns if 'MACDh' in c][0]
    df['MACD_Hist_Norm'] = df[macd_hist_col] / df['Close']
    
    df['Rolling_15m_Return'] = (df['Close'] - df['Close'].shift(15)) / df['Close'].shift(15)
    
    # 9-11. Volatility & Volume Features
    bb_pctb_col = [c for c in df.columns if 'BBP' in c][0]
    df['BB_PctB'] = df[bb_pctb_col]
    
    atr_col = [c for c in df.columns if 'ATR' in c][0]
    df['ATR_Norm'] = df[atr_col] / df['Close']
    
    df['RVOL'] = df['Volume'] / df['SMA_Volume_390']
    
    # 12-14. Context / Regime Features
    adx_col = [c for c in df.columns if 'ADX' in c and 'ADXR' not in c][0]
    df['ADX_14'] = df[adx_col]
    
    # Minutes from Open (9:30 AM local time for index, which is EST/EDT)
    market_open = df.index.floor('D') + pd.Timedelta(hours=9, minutes=30)
    df['Minutes_From_Open'] = ((df.index - market_open).total_seconds() // 60).astype(int)
    
    df['Day_Of_Week'] = df.index.dayofweek
    
    # --- Final Cleanup ---
    
    features_to_keep = [
        'Open', 'High', 'Low', 'Close', 'Volume', 
        '9EMA_Dist', '200EMA_Dist', 'EMA_Spread', 'VWAP_Dist',
        'RSI_14', 'WillR_14', 'MACD_Hist_Norm', 'Rolling_15m_Return',
        'BB_PctB', 'ATR_Norm', 'RVOL',
        'ADX_14', 'Minutes_From_Open', 'Day_Of_Week'
    ]
    
    df = df[features_to_keep]
    
    print(f"Shape before dropping NaNs: {df.shape}")
    df.dropna(inplace=True)
    print(f"Shape after dropping NaNs: {df.shape}")
    
    print(f"Saving features to {args.output}...")
    df.to_parquet(args.output, index=True)
    print("Feature generation complete!")

if __name__ == "__main__":
    main()
