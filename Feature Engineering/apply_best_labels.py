import optuna_labeling as ol
import numpy as np
import os

# Best parameters found from previous 200 trial run
best_params = {
    'method': 'LogReturn_StdDev',
    'long_tp_mult': 2.44,
    'long_sl_mult': 0.53,
    'short_tp_mult': 1.93,
    'short_sl_mult': 0.54
}

def main():
    print("Loading continuous 1m data...")
    df = ol.load_and_preprocess()
    
    obj = ol.TripleBarrierObjective(df)
    
    # Use LogReturn_StdDev array as base step
    d_base = obj.d_logret
    
    d_long_pnl = np.zeros(obj.n, dtype=np.float32)
    d_short_pnl = np.zeros(obj.n, dtype=np.float32)
    d_long_win = np.zeros(obj.n, dtype=np.int8)
    d_short_win = np.zeros(obj.n, dtype=np.int8)
    
    print(f"Applying Triple Barrier with lookahead {ol.LOOKAHEAD}...")
    ol.evaluate_barriers_parallel(
        obj.d_close, obj.d_high, obj.d_low, d_base,
        best_params['long_tp_mult'], best_params['long_sl_mult'], 
        best_params['short_tp_mult'], best_params['short_sl_mult'],
        ol.LOOKAHEAD, ol.POINT_VALUE,
        d_long_pnl, d_short_pnl, d_long_win, d_short_win
    )
    
    df['long_win'] = d_long_win
    df['short_win'] = d_short_win
    
    # The last `LOOKAHEAD` rows are invalid
    df = df.iloc[:-ol.LOOKAHEAD].copy()
    
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs', 'nq_labeled_base.parquet')
    print(f"Saving base labeled dataset with {len(df)} rows to {out_path}...")
    df.to_parquet(out_path)
    
    # Print quick stats
    print("\nTarget Label Distribution:")
    print(f"Long Wins: {df['long_win'].sum():,}")
    print(f"Short Wins: {df['short_win'].sum():,}")
    print("Done!")

if __name__ == "__main__":
    main()
