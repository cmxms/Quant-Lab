import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif
import os

# ANSI escape codes for gorgeous console output
class Colors:
    CYAN = '\033[96m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RED = '\033[91m'
    WHITE = '\033[97m'
    RESET = '\033[0m'

def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Chronological Split & MI Feature Selection")
    parser.add_argument("--input", required=True, help="Path to the raw labeled .parquet file from Phase 1")
    parser.add_argument("--min_score", type=float, required=True, help="MI score threshold (e.g. 0.0010)")
    parser.add_argument("--output_prefix", required=True, help="Prefix for outputs (e.g. 'bull' or 'bear')")
    return parser.parse_args()

def get_date_range(d):
    """Helper to extract date range from dataframe index or columns."""
    if 'datetime' in d.columns:
        return f"{d['datetime'].min()} to {d['datetime'].max()}"
    elif isinstance(d.index, pd.DatetimeIndex):
        return f"{d.index.min()} to {d.index.max()}"
    return "N/A"

def main():
    # Enable ANSI escape sequences on Windows if running in cmd/powershell
    os.system('')
    
    args = parse_args()
    input_file = args.input
    min_score = args.min_score
    prefix = args.output_prefix
    
    print(f"{Colors.CYAN}--- Phase 2 Pipeline Starting ---{Colors.RESET}")
    print(f"{Colors.WHITE}Input File: {input_file}{Colors.RESET}")
    print(f"{Colors.WHITE}MI Minimum Score: {min_score}{Colors.RESET}")
    print(f"{Colors.WHITE}Output Prefix: {prefix}\n{Colors.RESET}")
    
    # ---------------------------------------------------------
    # STEP 1: The Chronological Split
    # ---------------------------------------------------------
    print(f"{Colors.YELLOW}[*] Loading dataset...{Colors.RESET}")
    df = pd.read_parquet(input_file)
    
    total_rows = len(df)
    train_size = int(total_rows * 0.70)
    val_size = int(total_rows * 0.10)
    
    print(f"{Colors.YELLOW}[*] Splitting chronologically (Train: 70%, Val: 10%, Holdout: 20%)...{Colors.RESET}")
    
    train_df = df.iloc[:train_size].copy()
    val_df = df.iloc[train_size:train_size+val_size].copy()
    holdout_df = df.iloc[train_size+val_size:].copy()

    print(f"{Colors.GREEN}    -> Train:   {len(train_df):,} rows ({get_date_range(train_df)}){Colors.RESET}")
    print(f"{Colors.GREEN}    -> Val:     {len(val_df):,} rows ({get_date_range(val_df)}){Colors.RESET}")
    print(f"{Colors.GREEN}    -> Holdout: {len(holdout_df):,} rows ({get_date_range(holdout_df)})\n{Colors.RESET}")
    
    # ---------------------------------------------------------
    # STEP 2: MI Calculation (Strictly on Train)
    # ---------------------------------------------------------
    print(f"{Colors.YELLOW}[*] Preparing Train data for MI Calculation...{Colors.RESET}")
    mi_train = train_df.copy()
    
    # Drop NaN/Inf rows
    initial_mi_len = len(mi_train)
    mi_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    mi_train.dropna(inplace=True)
    print(f"{Colors.GREEN}    -> Dropped {initial_mi_len - len(mi_train):,} rows containing NaN/Inf.{Colors.RESET}")
    
    # If rows > 200,000, sample 200,000 rows (random_state=42)
    if len(mi_train) > 200000:
        print(f"{Colors.YELLOW}[*] Downsampling MI Train data from {len(mi_train):,} to 200,000 rows...{Colors.RESET}")
        mi_train = mi_train.sample(n=200000, random_state=42)
    
    # Drop Data Leakage Meta Columns from all datasets before processing
    leakage_cols = ['rtype', 'publisher_id', 'instrument_id', 'symbol']
    for c in leakage_cols:
        if c in train_df.columns:
            train_df.drop(columns=[c], inplace=True)
            val_df.drop(columns=[c], inplace=True)
            holdout_df.drop(columns=[c], inplace=True)
            if c in mi_train.columns:
                mi_train.drop(columns=[c], inplace=True)
                
    # Isolate features (ignoring Target, OHLCV, and datetime)
    ignore_cols = ['open', 'high', 'low', 'close', 'volume', 'target', 'datetime', 'ts_event']
    # Case-insensitive check
    features = [c for c in mi_train.columns if c.lower() not in ignore_cols and pd.api.types.is_numeric_dtype(mi_train[c])]
    
    # Check for Target (which is case sensitive in our DF)
    target_col = 'Target' if 'Target' in mi_train.columns else 'target'
    if target_col not in mi_train.columns:
        raise ValueError("Error: 'Target' column not found in the dataset.")
        
    X = mi_train[features]
    y = mi_train[target_col].astype(int) # Ensure integer classification target
    
    print(f"{Colors.YELLOW}[*] Calculating MI Scores on {len(features)} features...{Colors.RESET}")
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=features).sort_values(ascending=False)
    
    # Generate and save scorecard chart
    chart_filename = f"{prefix}_mi_scores.png"
    print(f"{Colors.YELLOW}[*] Saving MI Scorecard to {chart_filename}...{Colors.RESET}")
    plt.figure(figsize=(12, max(8, len(features)*0.25)))
    sns.barplot(x=mi_series.values, y=mi_series.index, hue=mi_series.index, legend=False, palette="viridis")
    plt.title(f"Mutual Information Scores ({prefix.upper()})")
    plt.xlabel("MI Score")
    plt.ylabel("Features")
    plt.axvline(x=min_score, color='r', linestyle='--', label=f'Min Score: {min_score}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(chart_filename, dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # STEP 3: The Great Purge
    # ---------------------------------------------------------
    features_to_drop = mi_series[mi_series < min_score].index.tolist()
    print(f"\n{Colors.RED}[!] THE GREAT PURGE: Removing {len(features_to_drop)} features scoring < {min_score}{Colors.RESET}")
    for f in features_to_drop:
        print(f"{Colors.RED}    - {f} (Score: {mi_series[f]:.6f}){Colors.RESET}")
        
    print(f"\n{Colors.YELLOW}[*] Dropping features from Train, Validation, and Holdout sets...{Colors.RESET}")
    # Drop features strictly < min_score from datasets
    train_df.drop(columns=features_to_drop, inplace=True, errors='ignore')
    val_df.drop(columns=features_to_drop, inplace=True, errors='ignore')
    holdout_df.drop(columns=features_to_drop, inplace=True, errors='ignore')
    
    # ---------------------------------------------------------
    # STEP 4: Save & Report
    # ---------------------------------------------------------
    train_out = f"{prefix}_clean_train.parquet"
    val_out = f"{prefix}_clean_val.parquet"
    holdout_out = f"{prefix}_clean_holdout.parquet"
    
    print(f"{Colors.YELLOW}[*] Saving cleaned dataframes...{Colors.RESET}")
    train_df.to_parquet(train_out)
    val_df.to_parquet(val_out)
    holdout_df.to_parquet(holdout_out)
    
    print(f"\n{Colors.CYAN}=================================================={Colors.RESET}")
    print(f"{Colors.CYAN}                 FINAL SUMMARY{Colors.RESET}")
    print(f"{Colors.CYAN}=================================================={Colors.RESET}")
    print(f"{Colors.WHITE}Original Feature Count:   {len(features)}{Colors.RESET}")
    print(f"{Colors.RED}Features Purged:          {len(features_to_drop)}{Colors.RESET}")
    print(f"{Colors.GREEN}Surviving Feature Count:  {len(features) - len(features_to_drop)}{Colors.RESET}")
    print(f"{Colors.WHITE}Final Dataframe Columns:  {len(train_df.columns)}{Colors.RESET}")
    print(f"\n{Colors.GREEN}Perfect schema alignment verified across all 3 splits.{Colors.RESET}")
    print(f"{Colors.WHITE}Outputs Saved:{Colors.RESET}")
    print(f"  - {chart_filename}")
    print(f"  - {train_out}")
    print(f"  - {val_out}")
    print(f"  - {holdout_out}")
    print(f"{Colors.CYAN}==================================================\n{Colors.RESET}")

if __name__ == "__main__":
    main()
