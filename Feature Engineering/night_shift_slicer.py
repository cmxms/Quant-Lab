"""
night_shift_slicer.py
=====================
Extracts the Overnight window (16:45 - 08:00 ET) from the full
pre-MI feature set and saves a Night Shift parquet.

Usage
-----
    python night_shift_slicer.py
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

# Day Shift window (what we EXCLUDE)
RTH_START_HOUR, RTH_START_MIN = 8, 0     # 08:00 ET
RTH_END_HOUR, RTH_END_MIN    = 16, 45    # 16:45 ET

DIV = "=" * 62

def main():
    print(f"\n  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}  NIGHT SHIFT SLICER  (Overnight Window){C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    input_path = _OUTPUT_DIR / "nq_features_ready.parquet"
    if not input_path.exists():
        print(f"  {C.RED}ERROR: {input_path} not found!{C.RESET}")
        sys.exit(1)

    print(f"\n  {C.CYAN}Loading raw feature set...{C.RESET}")
    df = pd.read_parquet(input_path)
    total_rows = len(df)
    print(f"    Loaded: {total_rows:,} rows x {df.shape[1]} columns")

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

    # Night = everything OUTSIDE 08:00 - 16:45 ET
    print(f"  {C.CYAN}Filtering to OUTSIDE 08:00 - 16:45 ET (Night Shift)...{C.RESET}")

    minutes_col = df[dt_col].dt.hour * 60 + df[dt_col].dt.minute
    rth_open  = RTH_START_HOUR * 60 + RTH_START_MIN   # 480
    rth_close = RTH_END_HOUR * 60 + RTH_END_MIN       # 1005

    # Night = before 08:00 OR after 16:45
    mask = (minutes_col < rth_open) | (minutes_col > rth_close)
    df_night = df[mask].copy()

    dropped = total_rows - len(df_night)
    print(f"    {C.RED}Day rows dropped:        {dropped:,}{C.RESET}")
    print(f"    {C.GREEN}Night Shift rows kept:   {len(df_night):,}{C.RESET}")
    print(f"    Retention: {len(df_night)/total_rows*100:.1f}%")

    df_night[dt_col] = df_night[dt_col].dt.tz_localize(None)

    print(f"\n    Date range: {df_night[dt_col].min()} to {df_night[dt_col].max()}")

    target = "Target" if "Target" in df_night.columns else "target"
    if target in df_night.columns:
        dist = df_night[target].value_counts().sort_index()
        print(f"    Target distribution:")
        for cls, cnt in dist.items():
            pct = cnt / len(df_night) * 100
            print(f"      Class {cls:>2}: {cnt:>10,}  ({pct:.1f}%)")

    out_path = _OUTPUT_DIR / "nq_night_shift_raw.parquet"
    df_night.to_parquet(out_path, index=False)
    size_mb = out_path.stat().st_size / 1_048_576
    print(f"\n  {C.GREEN}Saved: {out_path.name} ({size_mb:.1f} MB){C.RESET}")
    print(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}\n")

if __name__ == "__main__":
    main()
