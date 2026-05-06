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
def compute_triple_barrier_short(close_prices, high_prices, low_prices, atr_norm, lookahead=30, tp_mult=2.0, sl_mult=1.5):
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
        
        # For a Short position: 
        # Take Profit is hit when price DROPS by tp_mult * ATR
        # Stop Loss is hit when price RISES by sl_mult * ATR
        take_profit_barrier = current_close - (tp_mult * current_atr)
        stop_loss_barrier = current_close + (sl_mult * current_atr)
        
        hit = 0
        for j in range(1, lookahead + 1):
            idx = i + j
            curr_high = high_prices[idx]
            curr_low = low_prices[idx]
            
            if np.isnan(curr_high) or np.isnan(curr_low):
                continue
                
            hit_tp = curr_low <= take_profit_barrier
            hit_sl = curr_high >= stop_loss_barrier
            
            # Pessimistic Execution: If BOTH crossed in same candle, assume stop loss hit first
            if hit_sl and hit_tp:
                hit = -1
                break
            elif hit_sl:
                hit = -1
                break
            elif hit_tp:
                hit = 1
                break
                
        targets[i] = hit
        
    return targets


def main():
    parser = argparse.ArgumentParser(description="Generate ML targets for SHORT positions using the Triple Barrier Method.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input .parquet file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output labeled .parquet file.")
    parser.add_argument("--lookahead", type=int, default=30, help="Forward-looking window in rows (default: 30).")
    parser.add_argument("--tp_mult", type=float, default=2.0, help="Multiplier for the Take Profit barrier (default: 2.0).")
    parser.add_argument("--sl_mult", type=float, default=1.5, help="Multiplier for the Stop Loss barrier (default: 1.5).")
    
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

    print(f"Loaded {len(df):,} rows. Calculating Short-biased Triple Barrier targets...")
    
    # Extract numpy arrays for Numba (float64 is explicitly cast to ensure fastmath compatibility)
    close_prices = df['Close'].to_numpy(dtype=np.float64)
    high_prices = df['High'].to_numpy(dtype=np.float64)
    low_prices = df['Low'].to_numpy(dtype=np.float64)
    atr_norm = df['ATR_Norm'].to_numpy(dtype=np.float64)

    # Numba compilation happens on this first call
    targets = compute_triple_barrier_short(
        close_prices, 
        high_prices, 
        low_prices, 
        atr_norm, 
        lookahead=args.lookahead, 
        tp_mult=args.tp_mult, 
        sl_mult=args.sl_mult
    )
    
    df['Target'] = targets

    # Drop the rows at the very end that do not have a full forward window
    print(f"Dropping the last {args.lookahead} rows to ensure full forward window data integrity...")
    if args.lookahead > 0:
        df = df.iloc[:-args.lookahead].copy()

    print(f"Saving labeled data to {args.output}...")
    try:
        # Intentionally leaving index=False OUT so the timestamp index is preserved
        df.to_parquet(args.output, engine=parquet_engine)
        print("Success! Short-biased labeled data saved.")
        
        # Print basic label statistics
        target_counts = df['Target'].value_counts()
        print("\nTarget Distribution (Short Strategy):")
        print(f"  1 (Take Profit Hit): {target_counts.get(1, 0):,}")
        print(f" -1 (Stop Loss Hit):   {target_counts.get(-1, 0):,}")
        print(f"  0 (No Hit):          {target_counts.get(0, 0):,}")
        
    except Exception as e:
        print(f"Error saving parquet file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
