"""
day_shift_slicer.py
===================
Extracts the US Momentum window (08:00 - 16:45 ET) from the full
pre-MI feature set and saves a Day Shift parquet for re-evaluation.

Usage
-----
    python day_shift_slicer.py
"""

import os, sys
from pathlib import Path
import pandas as pd
import numpy as np

os.system('')

class C:
    CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'
    MAGENTA='\033[95m'; WHITE='\033[97m'; BOLD='\033[1m'; DIM='\033[2m'
    RESET='\033[0m'

_HERE = Path(__file__).parent.resolve()
_OUTPUT_DIR = _HERE / "Outputs"

# US Momentum window (ET)
RTH_START_HOUR, RTH_START_MIN = 8, 0     # 08:00 ET
RTH_END_HOUR, RTH_END_MIN    = 16, 45    # 16:45 ET

DIV = "=" * 62

def main():
    print(f"\n  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}  DAY SHIFT SLICER  (US Momentum Window){C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    # ── Load the pre-MI feature set ──────────────────────────────────────────
    input_path = _OUTPUT_DIR / "nq_features_ready.parquet"
    if not input_path.exists():
        print(f"  {C.RED}ERROR: {input_path} not found!{C.RESET}")
        sys.exit(1)

    print(f"\n  {C.CYAN}Loading raw feature set...{C.RESET}")
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    print(f"    Loaded: {total_rows:,} rows x {df.shape[1]} columns")

    # ── Extract datetime and convert to ET ───────────────────────────────────
    dt_col = None
    for c in ["ts_event", "datetime"]:
        if c in df.columns:
            dt_col = c
            break

    if dt_col is None:
        print(f"  {C.RED}ERROR: No datetime column found!{C.RESET}")
        sys.exit(1)

    print(f"  {C.CYAN}Converting '{dt_col}' to US/Eastern...{C.RESET}")
    df[dt_col] = pd.to_datetime(df[dt_col], utc=True).dt.tz_convert("US/Eastern")

    # ── Apply the US Momentum filter ─────────────────────────────────────────
    print(f"  {C.CYAN}Filtering to 08:00 - 16:45 ET...{C.RESET}")

    minutes_col = df[dt_col].dt.hour * 60 + df[dt_col].dt.minute
    rth_open  = RTH_START_HOUR * 60 + RTH_START_MIN   # 480
    rth_close = RTH_END_HOUR * 60 + RTH_END_MIN       # 1005

    mask = (minutes_col >= rth_open) & (minutes_col <= rth_close)
    df_day = df[mask].copy()

    dropped = total_rows - len(df_day)
    print(f"    {C.RED}Overnight rows dropped: {dropped:,}{C.RESET}")
    print(f"    {C.GREEN}Day Shift rows kept:    {len(df_day):,}{C.RESET}")
    print(f"    Retention: {len(df_day)/total_rows*100:.1f}%")

    # Strip timezone info before saving (parquet doesn't love tz-aware)
    df_day[dt_col] = df_day[dt_col].dt.tz_localize(None)

    # ── Date range ───────────────────────────────────────────────────────────
    print(f"\n    Date range: {df_day[dt_col].min()} to {df_day[dt_col].max()}")

    # ── Target distribution ──────────────────────────────────────────────────
    target = "Target" if "Target" in df_day.columns else "target"
    if target in df_day.columns:
        dist = df_day[target].value_counts().sort_index()
        print(f"    Target distribution:")
        for cls, cnt in dist.items():
            pct = cnt / len(df_day) * 100
            print(f"      Class {cls:>2}: {cnt:>10,}  ({pct:.1f}%)")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = _OUTPUT_DIR / "nq_day_shift_raw.parquet"
    df_day.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\n  {C.GREEN}Saved: {out_path.name} ({size_mb:.1f} MB){C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}\n")


if __name__ == "__main__":
    main()
