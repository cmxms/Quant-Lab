import os
import pandas as pd
import numpy as np
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from numba import njit, prange
import math
import warnings
warnings.filterwarnings("ignore")

# Configuration
PARQUET_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                            'Data Ingestion', 'Outputs', 'nq_continuous_1m.parquet')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LOOKAHEAD = 20
POINT_VALUE = 20.0
N_TRIALS = 200

# ==========================================
# 1. Numba Parallel CPU Kernel for Triple Barrier
# ==========================================
@njit(parallel=True, fastmath=True)
def evaluate_barriers_parallel(close, high, low, base_step, 
                               long_tp_mult, long_sl_mult, short_tp_mult, short_sl_mult, 
                               lookahead, point_value,
                               out_long_pnl, out_short_pnl, out_long_win, out_short_win):
    n = len(close)
    # Loop over all rows in parallel
    for i in prange(n - lookahead):
        c = close[i]
        step = base_step[i]
        
        if math.isnan(c) or math.isnan(step) or step <= 0:
            continue
            
        # Barriers
        long_tp = c + (step * long_tp_mult)
        long_sl = c - (step * long_sl_mult)
        
        short_tp = c - (step * short_tp_mult)
        short_sl = c + (step * short_sl_mult)
        
        long_resolved = False
        short_resolved = False
        
        for j in range(1, lookahead + 1):
            idx = i + j
            h = high[idx]
            l = low[idx]
            
            # Long
            if not long_resolved:
                hit_tp = h >= long_tp
                hit_sl = l <= long_sl
                if hit_tp and hit_sl:
                    out_long_pnl[i] = (long_sl - c) * point_value
                    long_resolved = True
                elif hit_sl:
                    out_long_pnl[i] = (long_sl - c) * point_value
                    long_resolved = True
                elif hit_tp:
                    bonus = 1.0 - (0.2 * ((j - 1) / max(1.0, float(lookahead - 1))))
                    out_long_pnl[i] = (long_tp - c) * point_value * bonus
                    out_long_win[i] = 1
                    long_resolved = True
                    
            # Short
            if not short_resolved:
                hit_tp = l <= short_tp
                hit_sl = h >= short_sl
                if hit_tp and hit_sl:
                    out_short_pnl[i] = (c - short_sl) * point_value
                    short_resolved = True
                elif hit_sl:
                    out_short_pnl[i] = (c - short_sl) * point_value
                    short_resolved = True
                elif hit_tp:
                    bonus = 1.0 - (0.2 * ((j - 1) / max(1.0, float(lookahead - 1))))
                    out_short_pnl[i] = (c - short_tp) * point_value * bonus
                    out_short_win[i] = 1
                    short_resolved = True
                    
            if long_resolved and short_resolved:
                break
                
        # Vertical Barrier
        if not long_resolved:
            out_long_pnl[i] = (close[i + lookahead] - c) * point_value
            if out_long_pnl[i] > 0:
                out_long_win[i] = 1
                
        if not short_resolved:
            out_short_pnl[i] = (c - close[i + lookahead]) * point_value
            if out_short_pnl[i] > 0:
                out_short_win[i] = 1

# ==========================================
# 2. Data Preprocessing
# ==========================================
def load_and_preprocess():
    print(f"Loading data from {PARQUET_PATH}...")
    df = pd.read_parquet(PARQUET_PATH)
    
    # Standardize datetime
    dt_col = 'ts_event' if 'ts_event' in df.columns else 'datetime'
    df[dt_col] = pd.to_datetime(df[dt_col])
    
    if df[dt_col].dt.tz is None:
        df[dt_col] = df[dt_col].dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
    else:
        df[dt_col] = df[dt_col].dt.tz_convert('US/Eastern')
        
    print("Categorizing Trading Sessions...")
    hours = df[dt_col].dt.hour
    
    conditions = [
        (hours >= 18) | (hours < 3),
        (hours >= 3) & (hours < 8),
        (hours >= 8) & (hours < 12),
        (hours >= 12) & (hours < 14),
        (hours >= 14) & (hours < 17)
    ]
    choices = ['Asian', 'London', 'US_AM', 'US_Lunch', 'US_PM']
    df['Session'] = np.select(conditions, choices, default='Post_Market')
    
    print("Calculating Rolling Metrics (ATR, LogRet, Fixed)...")
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = np.maximum(df['high'] - df['low'], 
               np.maximum(abs(df['high'] - df['prev_close']), 
                          abs(df['low'] - df['prev_close'])))
    df['ATR'] = df['tr'].rolling(100).mean()
    
    df['log_ret'] = np.log(df['close'] / df['prev_close'])
    df['LogReturn_StdDev'] = df['log_ret'].rolling(100).std() * df['close'] # Convert to points
    
    df['Fixed_Percentage'] = df['close'] * 0.001 # 0.1% of Close
    
    # Drop rows where rolling metrics are NaN
    df.dropna(subset=['ATR', 'LogReturn_StdDev', 'Fixed_Percentage'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    print(f"Data ready. Total rows: {len(df):,}")
    return df

# ==========================================
# 3. Optuna Orchestration
# ==========================================
class TripleBarrierObjective:
    def __init__(self, df):
        self.df = df
        
        # Pre-allocate numpy arrays
        self.d_close = df['close'].to_numpy(dtype=np.float32)
        self.d_high = df['high'].to_numpy(dtype=np.float32)
        self.d_low = df['low'].to_numpy(dtype=np.float32)
        
        self.d_atr = df['ATR'].to_numpy(dtype=np.float32)
        self.d_logret = df['LogReturn_StdDev'].to_numpy(dtype=np.float32)
        self.d_fixed = df['Fixed_Percentage'].to_numpy(dtype=np.float32)
        
        self.n = len(df)
        
        # Keep track of best trial info for heatmap
        self.best_expectancy = -np.inf
        self.best_long_win = None
        self.best_short_win = None
        
    def __call__(self, trial):
        method = trial.suggest_categorical('method', ['LogReturn_StdDev', 'Fixed_Percentage', 'ATR'])
        
        long_tp_mult = trial.suggest_float('long_tp_mult', 0.5, 3.0)
        long_sl_mult = trial.suggest_float('long_sl_mult', 0.5, 2.0)
        short_tp_mult = trial.suggest_float('short_tp_mult', 0.5, 3.0)
        short_sl_mult = trial.suggest_float('short_sl_mult', 0.5, 2.0)
        
        if method == 'ATR':
            d_base = self.d_atr
        elif method == 'LogReturn_StdDev':
            d_base = self.d_logret
        else:
            d_base = self.d_fixed
            
        d_long_pnl = np.zeros(self.n, dtype=np.float32)
        d_short_pnl = np.zeros(self.n, dtype=np.float32)
        d_long_win = np.zeros(self.n, dtype=np.int8)
        d_short_win = np.zeros(self.n, dtype=np.int8)
        
        # Run Numba CPU parallel kernel
        evaluate_barriers_parallel(
            self.d_close, self.d_high, self.d_low, d_base,
            long_tp_mult, long_sl_mult, short_tp_mult, short_sl_mult,
            LOOKAHEAD, POINT_VALUE,
            d_long_pnl, d_short_pnl, d_long_win, d_short_win
        )
        
        # Fetch results (already numpy arrays)
        long_pnl = d_long_pnl
        short_pnl = d_short_pnl

        
        # Maximize Combined Expectancy
        total_long_pnl = np.sum(long_pnl)
        total_short_pnl = np.sum(short_pnl)
        combined_expectancy = total_long_pnl + total_short_pnl
        
        # Save results if best
        if combined_expectancy > self.best_expectancy:
            self.best_expectancy = combined_expectancy
            self.best_long_win = d_long_win.copy()
            self.best_short_win = d_short_win.copy()
            
        return combined_expectancy

# ==========================================
# 4. Output & Heatmap
# ==========================================
def generate_heatmap(df, long_win, short_win):
    print("Generating Win Rate Heatmap across Sessions...")
    
    # Calculate valid trade count (excluding the last LOOKAHEAD rows)
    valid_mask = np.ones(len(df), dtype=bool)
    valid_mask[-LOOKAHEAD:] = False
    
    df['long_win'] = long_win
    df['short_win'] = short_win
    df['valid'] = valid_mask
    
    valid_df = df[df['valid'] & (df['Session'] != 'Post_Market')]
    
    # Calculate win rates
    heatmap_data = valid_df.groupby('Session')[['long_win', 'short_win']].mean().reset_index()
    heatmap_data.set_index('Session', inplace=True)
    
    # Reorder index logically
    order = ['Asian', 'London', 'US_AM', 'US_Lunch', 'US_PM']
    heatmap_data = heatmap_data.reindex(order)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(heatmap_data * 100, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Win Rate (%)'})
    plt.title('Win Rate by Trading Session (Best Trial)')
    plt.ylabel('Session')
    plt.xlabel('Direction')
    plt.tight_layout()
    
    heatmap_path = os.path.join(OUTPUT_DIR, 'session_win_rate_heatmap.png')
    plt.savefig(heatmap_path)
    print(f"Heatmap saved to {heatmap_path}")

def main():
    df = load_and_preprocess()
    
    print(f"\nInitializing Optuna Study for {N_TRIALS} trials...")
    study = optuna.create_study(direction='maximize')
    objective = TripleBarrierObjective(df)
    
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Peak Combined Expectancy: ${study.best_value:,.2f}")
    print(f"Winning Method: {study.best_params['method']}")
    print(f"Optimal Long TP Multiplier: {study.best_params['long_tp_mult']:.2f}")
    print(f"Optimal Long SL Multiplier: {study.best_params['long_sl_mult']:.2f}")
    print(f"Optimal Short TP Multiplier: {study.best_params['short_tp_mult']:.2f}")
    print(f"Optimal Short SL Multiplier: {study.best_params['short_sl_mult']:.2f}")
    
    generate_heatmap(df, objective.best_long_win, objective.best_short_win)

if __name__ == "__main__":
    main()
