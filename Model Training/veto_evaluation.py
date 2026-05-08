"""
veto_evaluation.py
==================
Post-Veto Performance Evaluation

Loads all saved models, applies the optimal thresholds, then drops any
trade the Meta-Model predicts as Is_Correct == 0 (The Veto).
Reports the before/after R-Expectancy comparison.
"""

import os, sys, logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.system('')  # enable ANSI on Windows

# ── ANSI ────────────────────────────────────────────────────────────────────
class C:
    CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'
    MAGENTA='\033[95m'; WHITE='\033[97m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Config ──────────────────────────────────────────────────────────────────
_HERE        = Path(__file__).parent.resolve()
_FEATURE_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_DIR   = _HERE / "Outputs"

TARGET_COL = "Target"
DROP_COLS  = ["Open","High","Low","Close","Volume",
              "Minutes_From_Open","Day_Of_Week","datetime","ts_event"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)
DIV = "=" * 62

def _latest(directory, pattern):
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"{C.RED}No file matching '{pattern}' in {directory}{C.RESET}")
        sys.exit(1)
    return matches[-1]

def main():
    log.info(f"\n{DIV}")
    log.info(f"  {C.BOLD}{C.MAGENTA}VETO EVALUATION — Before vs After Meta-Model{C.RESET}")
    log.info(DIV)

    # ── Load Models ──────────────────────────────────────────────────────────
    lgbm_model = joblib.load(_latest(_MODEL_DIR, "lgbm_model_*.joblib"))
    rf_model   = joblib.load(_latest(_MODEL_DIR, "rf_model_*.joblib"))
    mlp_model  = joblib.load(_latest(_MODEL_DIR, "mlp_model_*.joblib"))
    mlp_scaler = joblib.load(_latest(_MODEL_DIR, "mlp_scaler_*.joblib"))
    meta_model = joblib.load(_latest(_MODEL_DIR, "meta_model_*.joblib"))
    thresholds = joblib.load(_latest(_MODEL_DIR, "optimal_thresholds_*.joblib"))
    log.info(f"  {C.GREEN}All models & thresholds loaded.{C.RESET}")

    # ── Load Validation Data ─────────────────────────────────────────────────
    val_path = _latest(_FEATURE_DIR, "*val*")
    df = pd.read_parquet(val_path)
    target = TARGET_COL if TARGET_COL in df.columns else TARGET_COL.lower()
    y_val = df[target].to_numpy()
    X_val = df.drop(columns=[target])
    to_drop = [c for c in DROP_COLS if c in X_val.columns]
    if to_drop:
        X_val = X_val.drop(columns=to_drop)
    non_num = X_val.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X_val = X_val.drop(columns=non_num)
    log.info(f"  Validation set: {len(X_val):,} rows x {X_val.shape[1]} features")

    # ── R-Multiple parameters ────────────────────────────────────────────────
    LONG_WIN_R   = thresholds["long_win_r"]
    LONG_LOSS_R  = thresholds["long_loss_r"]
    SHORT_WIN_R  = thresholds["short_win_r"]
    SHORT_LOSS_R = thresholds["short_loss_r"]
    lt = thresholds["long_threshold"]
    st = thresholds["short_threshold"]

    # ── Generate Ensemble Probabilities ──────────────────────────────────────
    X_val_scaled = mlp_scaler.transform(X_val)
    proba_lgbm = lgbm_model.predict_proba(X_val)
    proba_rf   = rf_model.predict_proba(X_val)
    proba_mlp  = mlp_model.predict_proba(X_val_scaled)
    ensemble_proba = (proba_lgbm + proba_rf + proba_mlp) / 3.0

    ref_classes = lgbm_model.classes_
    long_idx  = int(np.where(ref_classes == 1)[0][0])
    short_idx = int(np.where(ref_classes == -1)[0][0])
    chop_idx  = int(np.where(ref_classes == 0)[0][0])

    long_probs  = ensemble_proba[:, long_idx]
    short_probs = ensemble_proba[:, short_idx]

    # ── Apply Thresholds (Pre-Veto) ──────────────────────────────────────────
    long_mask  = long_probs >= lt
    short_mask = short_probs >= st
    both = long_mask & short_mask
    long_signal  = long_mask.copy()
    short_signal = short_mask.copy()
    long_signal[both]  = long_probs[both] >= short_probs[both]
    short_signal[both] = short_probs[both] > long_probs[both]
    triggered = long_signal | short_signal

    direction = np.zeros(len(y_val), dtype=int)
    direction[long_signal]  = 1
    direction[short_signal] = -1

    # Pre-Veto stats
    t_idx = np.where(triggered)[0]
    pre_long_wins   = np.sum((direction[t_idx] == 1)  & (y_val[t_idx] == 1))
    pre_long_losses = np.sum((direction[t_idx] == 1)  & (y_val[t_idx] != 1))
    pre_short_wins  = np.sum((direction[t_idx] == -1) & (y_val[t_idx] == -1))
    pre_short_losses= np.sum((direction[t_idx] == -1) & (y_val[t_idx] != -1))
    pre_wins   = pre_long_wins + pre_short_wins
    pre_losses = pre_long_losses + pre_short_losses
    pre_trades = pre_wins + pre_losses
    pre_total_r = (pre_long_wins * LONG_WIN_R + pre_long_losses * LONG_LOSS_R +
                   pre_short_wins * SHORT_WIN_R + pre_short_losses * SHORT_LOSS_R)
    pre_avg_r  = pre_total_r / pre_trades if pre_trades > 0 else 0
    pre_wr     = pre_wins / pre_trades * 100 if pre_trades > 0 else 0

    # ── Build Meta-Features for Triggered Rows ───────────────────────────────
    X_triggered = X_val.iloc[t_idx].copy()
    X_triggered['ens_long_proba']  = ensemble_proba[t_idx, long_idx]
    X_triggered['ens_short_proba'] = ensemble_proba[t_idx, short_idx]
    X_triggered['ens_chop_proba']  = ensemble_proba[t_idx, chop_idx]
    X_triggered['ens_direction']   = direction[t_idx]

    # ── Apply the Veto ───────────────────────────────────────────────────────
    meta_pred = meta_model.predict(X_triggered)
    approved = meta_pred == 1  # Only keep trades the Meta-Model approves

    # Post-Veto stats
    approved_idx = t_idx[approved]
    post_long_wins   = np.sum((direction[approved_idx] == 1)  & (y_val[approved_idx] == 1))
    post_long_losses = np.sum((direction[approved_idx] == 1)  & (y_val[approved_idx] != 1))
    post_short_wins  = np.sum((direction[approved_idx] == -1) & (y_val[approved_idx] == -1))
    post_short_losses= np.sum((direction[approved_idx] == -1) & (y_val[approved_idx] != -1))
    post_wins   = post_long_wins + post_short_wins
    post_losses = post_long_losses + post_short_losses
    post_trades = post_wins + post_losses
    post_total_r = (post_long_wins * LONG_WIN_R + post_long_losses * LONG_LOSS_R +
                    post_short_wins * SHORT_WIN_R + post_short_losses * SHORT_LOSS_R)
    post_avg_r  = post_total_r / post_trades if post_trades > 0 else 0
    post_wr     = post_wins / post_trades * 100 if post_trades > 0 else 0

    vetoed = pre_trades - post_trades

    # ── Report ───────────────────────────────────────────────────────────────
    log.info("")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}         BEFORE vs AFTER META-MODEL VETO{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {'Metric':<28} {'PRE-VETO':>12}   {'POST-VETO':>12}")
    log.info(f"  {'-'*56}")
    log.info(f"  {'Trades Triggered':<28} {C.YELLOW}{pre_trades:>12,}{C.RESET}   {C.GREEN}{post_trades:>12,}{C.RESET}")
    log.info(f"  {'Trades Vetoed':<28} {'':>12}   {C.RED}{vetoed:>12,}{C.RESET}")
    log.info(f"  {'Wins':<28} {C.GREEN}{pre_wins:>12,}{C.RESET}   {C.GREEN}{post_wins:>12,}{C.RESET}")
    log.info(f"  {'Losses':<28} {C.RED}{pre_losses:>12,}{C.RESET}   {C.RED}{post_losses:>12,}{C.RESET}")
    log.info(f"  {'Win Rate':<28} {pre_wr:>11.2f}%   {post_wr:>11.2f}%")
    log.info(f"  {'Total R':<28} {C.GREEN}{pre_total_r:>11.2f}R{C.RESET}   {C.BOLD}{C.GREEN}{post_total_r:>11.2f}R{C.RESET}")
    log.info(f"  {'Avg R per Trade':<28} {pre_avg_r:>11.2f}R   {C.BOLD}{C.GREEN}{post_avg_r:>11.2f}R{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    # Delta summary
    r_delta = post_total_r - pre_total_r
    wr_delta = post_wr - pre_wr
    avgr_delta = post_avg_r - pre_avg_r
    log.info(f"\n  {C.BOLD}VETO IMPACT:{C.RESET}")
    color_r = C.GREEN if r_delta >= 0 else C.RED
    color_wr = C.GREEN if wr_delta >= 0 else C.RED
    color_ar = C.GREEN if avgr_delta >= 0 else C.RED
    log.info(f"    Total R Delta       : {color_r}{r_delta:+.2f}R{C.RESET}")
    log.info(f"    Win Rate Delta      : {color_wr}{wr_delta:+.2f}%{C.RESET}")
    log.info(f"    Avg R/Trade Delta   : {color_ar}{avgr_delta:+.2f}R{C.RESET}")
    log.info(f"    Trades Removed      : {vetoed:,} ({vetoed/pre_trades*100:.1f}% of signals)")
    log.info("")

if __name__ == "__main__":
    main()
