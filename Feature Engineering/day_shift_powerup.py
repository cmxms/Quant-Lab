"""
day_shift_powerup.py
====================
Injects RVOL, EMA_9_ZScore, Above_ORB, Below_ORB into the existing
nq_day_shift_raw.parquet and re-saves it. Then triggers MI re-scoring.
"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

os.system('')

C_CYAN='\033[96m'; C_GREEN='\033[92m'; C_YELLOW='\033[93m'
C_WHITE='\033[97m'; C_BOLD='\033[1m'; C_MAGENTA='\033[95m'; C_DIM='\033[2m'; C_RESET='\033[0m'
DIV = "=" * 62

_FEAT = Path(__file__).parent.resolve() / "Outputs"

def main():
    print(f"\n  {C_BOLD}{C_MAGENTA}{DIV}{C_RESET}")
    print(f"  {C_BOLD}{C_MAGENTA}   DAY SHIFT POWER-UP  (Feature Injection){C_RESET}")
    print(f"  {C_BOLD}{C_MAGENTA}{DIV}{C_RESET}")

    raw_path = _FEAT / "nq_day_shift_raw.parquet"
    if not raw_path.exists():
        print(f"  {C_YELLOW}ERROR: {raw_path} not found!{C_RESET}")
        sys.exit(1)

    print(f"\n  {C_CYAN}Loading nq_day_shift_raw.parquet...{C_RESET}")
    df = pd.read_parquet(raw_path)
    print(f"    Loaded: {len(df):,} rows x {df.shape[1]} columns")
    original_cols = set(df.columns)

    # Detect datetime column for ORB
    dt_col = None
    for c in ["ts_event", "datetime"]:
        if c in df.columns:
            dt_col = c
            break

    if dt_col:
        df[dt_col] = pd.to_datetime(df[dt_col])
        # If no timezone, assume ET already (day_shift_slicer stripped it)
        if df[dt_col].dt.tz is None:
            hours = df[dt_col].dt.hour
            minutes = df[dt_col].dt.minute
        else:
            et = df[dt_col].dt.tz_convert("US/Eastern")
            hours = et.dt.hour
            minutes = et.dt.minute
        time_min = hours * 60 + minutes
        dates = df[dt_col].dt.date
    else:
        print(f"  {C_YELLOW}No datetime column found -- ORB features disabled.{C_RESET}")

    # ── 1. RVOL ──────────────────────────────────────────────────────────────
    print(f"  {C_CYAN}Injecting RVOL...{C_RESET}")
    vol_col = 'volume' if 'volume' in df.columns else 'Volume'
    vol_ma20 = df[vol_col].rolling(20).mean()
    df['RVOL'] = df[vol_col] / vol_ma20

    # ── 2. EMA_9 Z-Score ─────────────────────────────────────────────────────
    print(f"  {C_CYAN}Injecting EMA_9_ZScore...{C_RESET}")
    close_col = 'close' if 'close' in df.columns else 'Close'
    ema9_col = None
    for c in df.columns:
        if 'ema_9' in c.lower() or c == 'EMA_9':
            ema9_col = c
            break
    if ema9_col:
        diff = df[close_col] - df[ema9_col]
        diff_mean = diff.rolling(20).mean()
        diff_std  = diff.rolling(20).std()
        df['EMA_9_ZScore'] = (diff - diff_mean) / diff_std
        print(f"    Using EMA column: {ema9_col}")
    else:
        print(f"    {C_YELLOW}EMA_9 column not found -- computing from close...{C_RESET}")
        ema9 = df[close_col].ewm(span=9, adjust=False).mean()
        diff = df[close_col] - ema9
        diff_mean = diff.rolling(20).mean()
        diff_std  = diff.rolling(20).std()
        df['EMA_9_ZScore'] = (diff - diff_mean) / diff_std

    # ── 3. ORB Bias ──────────────────────────────────────────────────────────
    if dt_col:
        print(f"  {C_CYAN}Injecting ORB Bias (09:30-10:00 ET)...{C_RESET}")
        high_col = 'high' if 'high' in df.columns else 'High'
        low_col  = 'low' if 'low' in df.columns else 'Low'

        orb_mask = (time_min >= 570) & (time_min < 600)  # 09:30-10:00
        df['_date'] = dates
        
        # Get ORB high/low per day
        orb_data = df.loc[orb_mask].groupby('_date').agg(
            ORB_High=(high_col, 'max'),
            ORB_Low=(low_col, 'min')
        )
        
        df = df.merge(orb_data, left_on='_date', right_index=True, how='left')
        
        # Forward fill within each day (for bars after 10:00)
        df['ORB_High'] = df.groupby('_date')['ORB_High'].ffill()
        df['ORB_Low']  = df.groupby('_date')['ORB_Low'].ffill()
        
        df['Above_ORB'] = (df[close_col] > df['ORB_High']).astype(np.int8)
        df['Below_ORB'] = (df[close_col] < df['ORB_Low']).astype(np.int8)
        
        # Fill NaN for pre-ORB bars (before 10:00 on each day)
        df['Above_ORB'] = df['Above_ORB'].fillna(0).astype(np.int8)
        df['Below_ORB'] = df['Below_ORB'].fillna(0).astype(np.int8)
        
        df.drop(columns=['_date', 'ORB_High', 'ORB_Low'], inplace=True)
        
        orb_above_pct = df['Above_ORB'].mean() * 100
        orb_below_pct = df['Below_ORB'].mean() * 100
        print(f"    Above ORB: {orb_above_pct:.1f}% of bars  |  Below ORB: {orb_below_pct:.1f}% of bars")

    # ── Summary ──────────────────────────────────────────────────────────────
    new_cols = set(df.columns) - original_cols
    print(f"\n  {C_GREEN}New features added: {sorted(new_cols)}{C_RESET}")
    print(f"  Final shape: {len(df):,} rows x {df.shape[1]} columns")

    # Handle any NaN from rolling
    nan_before = df.isna().sum().sum()
    df.dropna(inplace=True)
    nan_dropped = nan_before
    print(f"  Dropped {len(pd.read_parquet(raw_path)) - len(df):,} rows with NaN from new rolling features")

    # ── Save ─────────────────────────────────────────────────────────────────
    df.to_parquet(raw_path, index=False)
    size_mb = raw_path.stat().st_size / 1_048_576
    print(f"\n  {C_GREEN}Saved: {raw_path.name} ({size_mb:.1f} MB){C_RESET}")
    print(f"  {C_BOLD}{C_MAGENTA}{DIV}{C_RESET}\n")

if __name__ == "__main__":
    main()
