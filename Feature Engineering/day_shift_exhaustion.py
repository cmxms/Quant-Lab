"""
day_shift_exhaustion.py
=======================
Injects Dual %R Exhaustion features into nq_day_shift_raw.parquet.
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
    print(f"  {B}{M}   DAY SHIFT EXHAUSTION  (Dual %R Injection){R}")
    print(f"  {B}{M}{DIV}{R}")

    raw_path = _FEAT / "nq_day_shift_raw.parquet"
    df = pd.read_parquet(raw_path)
    print(f"\n  {C}Loaded: {len(df):,} rows x {df.shape[1]} cols{R}")
    original_cols = set(df.columns)

    # Detect column names
    high_col = 'high' if 'high' in df.columns else 'High'
    low_col  = 'low' if 'low' in df.columns else 'Low'
    close_col = 'close' if 'close' in df.columns else 'Close'

    # Drop any existing versions to avoid duplicates
    for c in ['Fast_R','Slow_R','R_Locked_OB','R_Locked_OS','R_Exhaustion_Short','R_Exhaustion_Long']:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)

    # Fast %R (21) and Slow %R (112)
    print(f"  {C}Computing Fast_R (21) and Slow_R (112)...{R}")
    df['Fast_R'] = ta.willr(df[high_col], df[low_col], df[close_col], length=21)
    df['Slow_R'] = ta.willr(df[high_col], df[low_col], df[close_col], length=112)

    # Locked zones
    print(f"  {C}Computing R_Locked_OB / R_Locked_OS...{R}")
    df['R_Locked_OB'] = ((df['Fast_R'] >= -20) & (df['Slow_R'] >= -20)).astype(np.int8)
    df['R_Locked_OS'] = ((df['Fast_R'] <= -80) & (df['Slow_R'] <= -80)).astype(np.int8)

    # Exhaustion signals
    print(f"  {C}Computing R_Exhaustion_Short / R_Exhaustion_Long...{R}")
    df['R_Exhaustion_Short'] = ((df['R_Locked_OB'].shift(1) == 1) & (df['R_Locked_OB'] == 0)).astype(np.int8)
    df['R_Exhaustion_Long']  = ((df['R_Locked_OS'].shift(1) == 1) & (df['R_Locked_OS'] == 0)).astype(np.int8)

    # Stats
    new_cols = set(df.columns) - original_cols
    print(f"\n  {G}New features: {sorted(new_cols)}{R}")

    ob_pct = df['R_Locked_OB'].mean() * 100
    os_pct = df['R_Locked_OS'].mean() * 100
    ex_s = df['R_Exhaustion_Short'].sum()
    ex_l = df['R_Exhaustion_Long'].sum()
    print(f"    Locked OB: {ob_pct:.1f}% of bars  |  Locked OS: {os_pct:.1f}% of bars")
    print(f"    Exhaustion Short triggers: {ex_s:,}  |  Exhaustion Long triggers: {ex_l:,}")

    # Drop NaN from warmup
    before = len(df)
    df.dropna(inplace=True)
    print(f"    Dropped {before - len(df):,} NaN rows from %R warmup")

    # Save
    df.to_parquet(raw_path, index=False)
    size_mb = raw_path.stat().st_size / 1_048_576
    print(f"\n  {G}Saved: {raw_path.name} ({size_mb:.1f} MB)  |  {len(df):,} rows x {df.shape[1]} cols{R}")
    print(f"  {B}{M}{DIV}{R}\n")

if __name__ == "__main__":
    main()
