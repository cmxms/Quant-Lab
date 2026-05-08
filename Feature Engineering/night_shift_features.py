"""
night_shift_features.py
=======================
Injects RVOL, EMA_9_ZScore, and Dual %R Exhaustion into nq_night_shift_raw.parquet.
ORB features are NOT added (no opening range during overnight hours).
"""
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import pandas_ta as ta

os.system('')
C='\033[96m'; G='\033[92m'; Y='\033[93m'; B='\033[1m'; M='\033[95m'; R='\033[0m'
DIV = "=" * 62
_FEAT = Path(__file__).parent.resolve() / "Outputs"

def main():
    print(f"\n  {B}{M}{DIV}{R}")
    print(f"  {B}{M}   NIGHT SHIFT FEATURE INJECTION{R}")
    print(f"  {B}{M}{DIV}{R}")

    raw_path = _FEAT / "nq_night_shift_raw.parquet"
    df = pd.read_parquet(raw_path)
    print(f"\n  {C}Loaded: {len(df):,} rows x {df.shape[1]} cols{R}")
    original_cols = set(df.columns)

    close_col = 'close' if 'close' in df.columns else 'Close'
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col  = 'low' if 'low' in df.columns else 'Low'
    vol_col  = 'volume' if 'volume' in df.columns else 'Volume'

    # 1. RVOL
    print(f"  {C}Injecting RVOL...{R}")
    df['RVOL'] = df[vol_col] / df[vol_col].rolling(20).mean()

    # 2. EMA_9 Z-Score
    print(f"  {C}Injecting EMA_9_ZScore...{R}")
    ema9_col = None
    for c in df.columns:
        if 'ema_9' in c.lower() or c == 'EMA_9':
            ema9_col = c; break
    if ema9_col:
        diff = df[close_col] - df[ema9_col]
    else:
        diff = df[close_col] - df[close_col].ewm(span=9, adjust=False).mean()
    diff_mean = diff.rolling(20).mean()
    diff_std  = diff.rolling(20).std()
    df['EMA_9_ZScore'] = (diff - diff_mean) / diff_std

    # 3. Dual %R Exhaustion
    print(f"  {C}Injecting Dual %R (Fast=21, Slow=112)...{R}")
    df['Fast_R'] = ta.willr(df[high_col], df[low_col], df[close_col], length=21)
    df['Slow_R'] = ta.willr(df[high_col], df[low_col], df[close_col], length=112)

    df['R_Locked_OB'] = ((df['Fast_R'] >= -20) & (df['Slow_R'] >= -20)).astype(np.int8)
    df['R_Locked_OS'] = ((df['Fast_R'] <= -80) & (df['Slow_R'] <= -80)).astype(np.int8)

    df['R_Exhaustion_Short'] = ((df['R_Locked_OB'].shift(1) == 1) & (df['R_Locked_OB'] == 0)).astype(np.int8)
    df['R_Exhaustion_Long']  = ((df['R_Locked_OS'].shift(1) == 1) & (df['R_Locked_OS'] == 0)).astype(np.int8)

    new_cols = set(df.columns) - original_cols
    print(f"\n  {G}New features: {sorted(new_cols)}{R}")

    ob_pct = df['R_Locked_OB'].mean() * 100
    os_pct = df['R_Locked_OS'].mean() * 100
    print(f"    Locked OB: {ob_pct:.1f}%  |  Locked OS: {os_pct:.1f}%")
    print(f"    Exhaustion Short: {df['R_Exhaustion_Short'].sum():,}  |  Long: {df['R_Exhaustion_Long'].sum():,}")

    before = len(df)
    df.dropna(inplace=True)
    print(f"    Dropped {before - len(df):,} NaN rows")

    df.to_parquet(raw_path, index=False)
    size_mb = raw_path.stat().st_size / 1_048_576
    print(f"\n  {G}Saved: {raw_path.name} ({size_mb:.1f} MB)  |  {len(df):,} rows x {df.shape[1]} cols{R}")
    print(f"  {B}{M}{DIV}{R}\n")

if __name__ == "__main__":
    main()
