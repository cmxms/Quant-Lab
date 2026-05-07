import argparse
import sys
import os

try:
    import pandas as pd
except ImportError:
    print("Error: pandas is required. Please install it using 'pip install pandas'")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: numpy is required. Please install it using 'pip install numpy'")
    sys.exit(1)

try:
    from numba import njit
except ImportError:
    print("Error: numba is required. Please install it using 'pip install numba'")
    sys.exit(1)

try:
    # Check if either pyarrow or fastparquet is available
    import pyarrow
    parquet_engine = 'pyarrow'
except ImportError:
    try:
        import fastparquet
        parquet_engine = 'fastparquet'
    except ImportError:
        print("Error: Either 'pyarrow' or 'fastparquet' is required to read/write Parquet files.")
        print("Please install one, e.g., 'pip install pyarrow'")
        sys.exit(1)


@njit(fastmath=True)
def compute_triple_barrier(close_prices, high_prices, low_prices, atr_norm, lookahead=30, upper_mult=6.0, lower_mult=4.0):
    n = len(close_prices)
    targets = np.zeros(n, dtype=np.int8)
    
    # Process up to n - lookahead rows.
    # Rows after (n - lookahead) won't be processed and will stay 0.
    # We will drop those invalid rows later in the Pandas DataFrame.
    for i in range(n - lookahead):
        current_close = close_prices[i]
        
        # Handle potential NaNs to prevent math errors
        if np.isnan(current_close) or np.isnan(atr_norm[i]):
            continue
            
        current_atr = atr_norm[i] * current_close
        
        upper_barrier = current_close + (upper_mult * current_atr)
        lower_barrier = current_close - (lower_mult * current_atr)
        
        hit = 0
        for j in range(1, lookahead + 1):
            idx = i + j
            curr_high = high_prices[idx]
            curr_low = low_prices[idx]
            
            if np.isnan(curr_high) or np.isnan(curr_low):
                continue
                
            hit_upper = curr_high >= upper_barrier
            hit_lower = curr_low <= lower_barrier
            
            # Pessimistic Execution: If BOTH crossed in same candle, assume stop loss hit
            if hit_upper and hit_lower:
                hit = -1
                break
            elif hit_lower:
                hit = -1
                break
            elif hit_upper:
                hit = 1
                break
                
        targets[i] = hit
        
    return targets


def main():
    parser = argparse.ArgumentParser(description="Generate ML targets using the Triple Barrier Method.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .parquet file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output labeled .parquet file.")
    parser.add_argument("--lookahead", type=int, default=30, help="Forward-looking window in rows (default: 30).")
    parser.add_argument("--upper_mult", type=float, default=6.0, help="Multiplier for the Upper Barrier (default: 6.0).")
    parser.add_argument("--lower_mult", type=float, default=4.0, help="Multiplier for the Lower Barrier (default: 4.0).")
    
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print(f"Loading data from {args.input} using {parquet_engine}...")
    try:
        df = pd.read_parquet(args.input, engine=parquet_engine)
    except Exception as e:
        print(f"Error reading parquet file: {e}")
        sys.exit(1)

    required_columns = ['Close', 'High', 'Low', 'ATR_Norm']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        print(f"Error: The following required columns are missing from the dataset: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)

    print(f"Loaded {len(df):,} rows. Calculating Triple Barrier targets...")
    
    # Extract numpy arrays for Numba (float64 is explicitly cast to ensure fastmath compatibility)
    close_prices = df['Close'].to_numpy(dtype=np.float64)
    high_prices = df['High'].to_numpy(dtype=np.float64)
    low_prices = df['Low'].to_numpy(dtype=np.float64)
    atr_norm = df['ATR_Norm'].to_numpy(dtype=np.float64)

    # Numba compilation happens on this first call
    targets = compute_triple_barrier(
        close_prices, 
        high_prices, 
        low_prices, 
        atr_norm, 
        lookahead=args.lookahead, 
        upper_mult=args.upper_mult, 
        lower_mult=args.lower_mult
    )
    
    df['Target'] = targets

    # Drop the rows at the very end that do not have a full forward window
    print(f"Dropping the last {args.lookahead} rows to ensure full forward window data integrity...")
    if args.lookahead > 0:
        df = df.iloc[:-args.lookahead].copy()

    print(f"Saving labeled data to {args.output}...")
    try:
        df.to_parquet(args.output, engine=parquet_engine)
        print("Success! Labeled data saved.")
        
        # Print basic label statistics
        target_counts = df['Target'].value_counts()
        print("\nTarget Distribution:")
        print(f"  1 (Take Profit):  {target_counts.get(1, 0):,}")
        print(f" -1 (Stop Loss):    {target_counts.get(-1, 0):,}")
        print(f"  0 (No Hit):       {target_counts.get(0, 0):,}")
        
    except Exception as e:
        print(f"Error saving parquet file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
