import subprocess
import os
import sys
import pandas as pd
from datetime import datetime

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPDATER_SCRIPT = os.path.join(BASE_DIR, 'Raw', 'update_nq_data.py')
STITCHER_SCRIPT = os.path.join(BASE_DIR, 'rollover_stitch.py')

# Data Files
RAW_PARQUET = os.path.join(BASE_DIR, 'Raw', 'glbx-mdp3-20210505-20260505.ohlcv-1m.parquet')
CONTINUOUS_PARQUET = os.path.join(BASE_DIR, 'Outputs', 'nq_continuous_1m.parquet')

def run_step(name, command):
    print(f"\n--- Starting Step: {name} ---")
    try:
        # Use sys.executable to ensure we use the same python environment
        result = subprocess.run([sys.executable] + command, check=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"--- Step {name} Completed Successfully ---")
    except subprocess.CalledProcessError as e:
        print(f"Error during {name}:")
        print(e.stderr)
        sys.exit(1)

def main():
    print(f"NQ Pipeline Orchestrator - Started at {datetime.now()}")

    # Step 1: Update Raw Data (UTC)
    # This script downloads from yfinance and appends to the raw parquet
    run_step("Update Raw Data", [UPDATER_SCRIPT])

    # Step 2: Run Rollover Stitch (Raw UTC -> Continuous EST)
    # This script cleans duplicates, handles rollovers, back-adjusts, and converts to US/Eastern
    run_step("Stitch Continuous Series", [
        STITCHER_SCRIPT, 
        "--input", RAW_PARQUET, 
        "--output", CONTINUOUS_PARQUET
    ])

    # Step 3: Final Verification
    print("\n--- Final Pipeline Audit ---")
    if os.path.exists(CONTINUOUS_PARQUET):
        df = pd.read_parquet(CONTINUOUS_PARQUET)
        last_ts = df['ts_event'].iloc[-1]
        
        # Check timezone of last_ts
        # The stitcher converts to US/Eastern
        print(f"Success: Continuous data updated to {last_ts}")
        print(f"Total Rows in Continuous Series: {len(df)}")
        print(f"Destination: {CONTINUOUS_PARQUET}")
    else:
        print("Error: Continuous parquet file was not created.")

if __name__ == "__main__":
    main()
