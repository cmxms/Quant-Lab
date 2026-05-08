import argparse
import os
import sys
import pandas as pd
import numpy as np
import pandas_ta as ta

# ANSI Color Codes
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def getWeights(d, size):
    w = [1.]
    for k in range(1, size):
        w_ = -w[-1] / k * (d - k + 1)
        w.append(w_)
    return np.array(w)

def fast_fracdiff(series, d, window=100):
    weights = getWeights(d, window)
    res = np.convolve(series.to_numpy(), weights, mode='full')[:len(series)]
    res[:window-1] = np.nan
    return pd.Series(res, index=series.index)

def main():
    parser = argparse.ArgumentParser(description="Phase 2: NQ Algorithmic Feature Factory")
    parser.add_argument("--input", required=True, help="Path to input labeled Parquet file")
    parser.add_argument("--output", required=True, help="Path to output engineered Parquet file")
    parser.add_argument("--vix", required=False, help="Path to optional VIX 1-minute Parquet file")
    parser.add_argument("--tick", required=False, help="Path to optional NYSE TICK 1-minute Parquet file")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"{Colors.FAIL}Error: Input file '{args.input}' not found.{Colors.ENDC}")
        sys.exit(1)

    print(f"{Colors.OKCYAN}{Colors.BOLD}--- NQ FEATURE FACTORY ---{Colors.ENDC}")
    
    # 1. Data Ingestion & Target Standardization
    print(f"{Colors.OKBLUE}Loading Base Labeled Data...{Colors.ENDC}")
    df = pd.read_parquet(args.input)
    
    dt_col = 'ts_event' if 'ts_event' in df.columns else 'datetime'
    if dt_col in df.columns:
        df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert('US/Eastern')
        df.set_index(dt_col, inplace=True)
    
    df.sort_index(inplace=True)
    
    # Target Standardization (Pessimistic Collision Handling)
    print(f"{Colors.OKBLUE}Standardizing Targets...{Colors.ENDC}")
    df['Target'] = 0
    df.loc[(df['long_win'] == 1) & (df['short_win'] == 0), 'Target'] = 1
    df.loc[(df['short_win'] == 1) & (df['long_win'] == 0), 'Target'] = -1
    
    # Drop intermediate label columns
    cols_to_drop = [c for c in ['long_win', 'short_win', 'long_pnl', 'short_pnl'] if c in df.columns]
    df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

    # 2. Fractional Differencing
    print(f"{Colors.OKBLUE}Calculating Vectorized Fractional Differencing (d=0.45)...{Colors.ENDC}")
    df['Close_FracDiff'] = fast_fracdiff(df['close'], 0.45, window=100)

    # 3. Pandas-TA "Kitchen Sink"
    print(f"{Colors.OKBLUE}Applying Pandas-TA Technical Indicators...{Colors.ENDC}")
    
    # Standardize column names for pandas-ta
    df.rename(columns={c: c.lower() for c in df.columns if c != 'Target'}, inplace=True)
    
    # Apply indicators individually to avoid Strategy object issues in beta pandas_ta
    df.ta.rsi(length=14, append=True)
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    df.ta.cci(length=20, append=True)
    df.ta.willr(length=14, append=True)
    df.ta.ema(length=9, append=True)
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.natr(length=14, append=True)
    df.ta.obv(append=True)
    df.ta.vwap(append=True)
    df.ta.cmf(length=20, append=True)

    # 3b. Momentum Power-Up Features
    print(f"{Colors.OKBLUE}Engineering Momentum Power-Up Features...{Colors.ENDC}")

    # RVOL (Relative Volume)
    vol_ma20 = df['volume'].rolling(20).mean()
    df['RVOL'] = df['volume'] / vol_ma20

    # EMA_9 Z-Score
    ema9_col = [c for c in df.columns if 'ema_9' in c.lower() or c == 'EMA_9']
    if ema9_col:
        diff = df['close'] - df[ema9_col[0]]
        diff_mean = diff.rolling(20).mean()
        diff_std  = diff.rolling(20).std()
        df['EMA_9_ZScore'] = (diff - diff_mean) / diff_std

    # ORB Bias (Opening Range Breakout: 09:30-10:00 ET)
    df['_date'] = df.index.date
    df['_time_min'] = df.index.hour * 60 + df.index.minute
    orb_mask = (df['_time_min'] >= 570) & (df['_time_min'] < 600)  # 09:30-10:00
    orb_highs = df.loc[orb_mask].groupby('_date')['high'].transform('max')
    orb_lows  = df.loc[orb_mask].groupby('_date')['low'].transform('min')
    df['ORB_High'] = np.nan
    df['ORB_Low']  = np.nan
    df.loc[orb_mask, 'ORB_High'] = orb_highs
    df.loc[orb_mask, 'ORB_Low']  = orb_lows
    # Forward-fill ORB levels within each day
    df['ORB_High'] = df.groupby('_date')['ORB_High'].ffill()
    df['ORB_Low']  = df.groupby('_date')['ORB_Low'].ffill()
    df['Above_ORB'] = (df['close'] > df['ORB_High']).astype(np.int8)
    df['Below_ORB'] = (df['close'] < df['ORB_Low']).astype(np.int8)
    df.drop(columns=['_date', '_time_min', 'ORB_High', 'ORB_Low'], inplace=True)

    # 3c. Dual %R Exhaustion Features
    print(f"{Colors.OKBLUE}Engineering Dual %R Exhaustion Features...{Colors.ENDC}")

    # Fast %R (21) and Slow %R (112)
    df['Fast_R'] = ta.willr(df['high'], df['low'], df['close'], length=21)
    df['Slow_R'] = ta.willr(df['high'], df['low'], df['close'], length=112)

    # Locked zones (both %R agree on overbought/oversold)
    df['R_Locked_OB'] = ((df['Fast_R'] >= -20) & (df['Slow_R'] >= -20)).astype(np.int8)
    df['R_Locked_OS'] = ((df['Fast_R'] <= -80) & (df['Slow_R'] <= -80)).astype(np.int8)

    # Exhaustion signals (locked zone just broke)
    df['R_Exhaustion_Short'] = ((df['R_Locked_OB'].shift(1) == 1) & (df['R_Locked_OB'] == 0)).astype(np.int8)
    df['R_Exhaustion_Long']  = ((df['R_Locked_OS'].shift(1) == 1) & (df['R_Locked_OS'] == 0)).astype(np.int8)

    # 4. Market Internals Injection
    if args.vix:
        if os.path.exists(args.vix):
            print(f"{Colors.OKBLUE}Injecting VIX Market Internals...{Colors.ENDC}")
            vix_df = pd.read_parquet(args.vix)
            # Ensure standard datetime index
            vdt = 'ts_event' if 'ts_event' in vix_df.columns else ('datetime' if 'datetime' in vix_df.columns else None)
            if vdt:
                vix_df[vdt] = pd.to_datetime(vix_df[vdt], utc=True).dt.tz_convert('US/Eastern')
                vix_df.set_index(vdt, inplace=True)
            vix_df = vix_df.add_prefix('VIX_')
            df = df.join(vix_df, how='left')
        else:
            print(f"{Colors.WARNING}Warning: VIX file '{args.vix}' not found.{Colors.ENDC}")

    if args.tick:
        if os.path.exists(args.tick):
            print(f"{Colors.OKBLUE}Injecting NYSE TICK Market Internals...{Colors.ENDC}")
            tick_df = pd.read_parquet(args.tick)
            tdt = 'ts_event' if 'ts_event' in tick_df.columns else ('datetime' if 'datetime' in tick_df.columns else None)
            if tdt:
                tick_df[tdt] = pd.to_datetime(tick_df[tdt], utc=True).dt.tz_convert('US/Eastern')
                tick_df.set_index(tdt, inplace=True)
            tick_df = tick_df.add_prefix('TICK_')
            df = df.join(tick_df, how='left')
        else:
            print(f"{Colors.WARNING}Warning: TICK file '{args.tick}' not found.{Colors.ENDC}")
            
    if args.vix or args.tick:
        # Forward fill the injected data for minute-mismatches
        df.ffill(inplace=True)

    # 5. Categorical Encoding
    print(f"{Colors.OKBLUE}Applying Categorical Encoding...{Colors.ENDC}")
    if 'session' in df.columns:
        df = pd.get_dummies(df, columns=['session'], prefix='Session', drop_first=False)
        # Convert boolean dummies to int8
        for col in df.columns:
            if col.startswith('Session_'):
                df[col] = df[col].astype(np.int8)
                
    df['hour'] = df.index.hour.astype(np.int8)
    df['day_of_week'] = df.index.dayofweek.astype(np.int8)

    # 6. Cleanup & Output
    print(f"{Colors.OKBLUE}Cleaning Dataset (Dropping Warmup NaNs)...{Colors.ENDC}")
    initial_rows = len(df)
    df.dropna(inplace=True)
    final_rows = len(df)
    print(f"{Colors.OKGREEN}Dropped {initial_rows - final_rows:,} warmup rows.{Colors.ENDC}")

    # Reset index for saving to Parquet safely
    df.reset_index(inplace=True)
    
    print(f"{Colors.OKGREEN}Saving {final_rows:,} engineered rows to {args.output}...{Colors.ENDC}")
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_parquet(args.output, index=False)
    
    print(f"{Colors.OKCYAN}{Colors.BOLD}--- Feature Factory Complete! ---{Colors.ENDC}")

if __name__ == "__main__":
    main()
