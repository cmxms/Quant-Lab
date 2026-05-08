"""
vault_backtester.py
===================
Phase 4: Walk-Forward Out-of-Sample Backtest

This script performs a BLIND evaluation on the completely unseen holdout
dataset. No training, no threshold optimization — purely loading saved
Phase 3 artifacts and measuring real out-of-sample performance.

Pipeline Stage : Walk-Forward Testing (Phase 4)
Inputs         : nq_final_clean_holdout.parquet
                 All Phase 3 artifacts from Model Training/Outputs/
Outputs        : Console scorecard only (no model mutation)

Usage
-----
    python vault_backtester.py
    python vault_backtester.py --prefix day_shift
"""

import argparse
import os, sys, logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

os.system('')  # enable ANSI on Windows

# ── ANSI ────────────────────────────────────────────────────────────────────
class C:
    CYAN='\033[96m'; GREEN='\033[92m'; YELLOW='\033[93m'; RED='\033[91m'
    MAGENTA='\033[95m'; WHITE='\033[97m'; BOLD='\033[1m'; DIM='\033[2m'
    RESET='\033[0m'

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
THIN = "-" * 58

def _latest(directory, pattern):
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"{C.RED}No file matching '{pattern}' in {directory}{C.RESET}")
        sys.exit(1)
    return matches[-1]

STARTING_BALANCE = 50_000.00
RISK_PER_TRADE   = 50.00

def main():
    parser = argparse.ArgumentParser(description="Vault Backtester - Phase 4")
    parser.add_argument("--prefix", type=str, default="",
                        help="Regime prefix (e.g. 'day_shift').")
    args = parser.parse_args()

    prefix = args.prefix
    mp = f"{prefix}_" if prefix else ""
    holdout_substr = f"{prefix}_clean_holdout" if prefix else "holdout"
    regime_label = prefix.replace('_', ' ').title() if prefix else "Default"

    log.info("")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}   THE VAULT  ({regime_label} Regime){C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.DIM}No training. No optimization. Blind out-of-sample only.{C.RESET}")

    # ── Step 1: Environment Setup ────────────────────────────────────────────
    log.info(f"\n  {C.CYAN}[Step 1/4] Loading Phase 3 Artifacts (prefix='{mp}')...{C.RESET}")

    lgbm_model = joblib.load(_latest(_MODEL_DIR, f"{mp}lgbm_model_*.joblib"))
    log.info(f"    Loaded {mp}lgbm_model")
    rf_model   = joblib.load(_latest(_MODEL_DIR, f"{mp}rf_model_*.joblib"))
    log.info(f"    Loaded {mp}rf_model")
    mlp_model  = joblib.load(_latest(_MODEL_DIR, f"{mp}mlp_model_*.joblib"))
    log.info(f"    Loaded {mp}mlp_model")
    mlp_scaler = joblib.load(_latest(_MODEL_DIR, f"{mp}mlp_scaler_*.joblib"))
    log.info(f"    Loaded {mp}mlp_scaler")
    meta_model = joblib.load(_latest(_MODEL_DIR, f"{mp}meta_model_*.joblib"))
    log.info(f"    Loaded {mp}meta_model")
    thresholds = joblib.load(_latest(_MODEL_DIR, f"{mp}optimal_thresholds_*.joblib"))
    log.info(f"    Loaded {mp}optimal_thresholds")

    log.info(f"\n  {C.CYAN}Loading UNSEEN holdout data...{C.RESET}")
    holdout_path = _latest(_FEATURE_DIR, f"*{holdout_substr}*")
    df = pd.read_parquet(holdout_path)
    log.info(f"    File: {holdout_path.name}")

    target = TARGET_COL if TARGET_COL in df.columns else TARGET_COL.lower()
    if target not in df.columns:
        log.error(f"  {C.RED}Target column not found!{C.RESET}")
        sys.exit(1)

    y_holdout = df[target].to_numpy()
    X_holdout = df.drop(columns=[target])

    to_drop = [c for c in DROP_COLS if c in X_holdout.columns]
    if to_drop:
        X_holdout = X_holdout.drop(columns=to_drop)
    non_num = X_holdout.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_num:
        X_holdout = X_holdout.drop(columns=non_num)

    log.info(f"    Holdout set: {C.BOLD}{len(X_holdout):,}{C.RESET} rows x {X_holdout.shape[1]} features")
    log.info(f"    Target dist: {dict(zip(*np.unique(y_holdout, return_counts=True)))}")

    # R-Multiple parameters (loaded from saved thresholds)
    LONG_WIN_R   = thresholds["long_win_r"]
    LONG_LOSS_R  = thresholds["long_loss_r"]
    SHORT_WIN_R  = thresholds["short_win_r"]
    SHORT_LOSS_R = thresholds["short_loss_r"]
    lt = thresholds["long_threshold"]
    st = thresholds["short_threshold"]

    log.info(f"\n    {C.WHITE}Loaded Thresholds  -> Long: {C.CYAN}{lt:.2f}{C.RESET}  |  Short: {C.CYAN}{st:.2f}{C.RESET}")
    log.info(f"    {C.WHITE}R-Multiples        -> LW: +{LONG_WIN_R:.2f}R  LL: {LONG_LOSS_R:.2f}R  SW: +{SHORT_WIN_R:.2f}R  SL: {SHORT_LOSS_R:.2f}R{C.RESET}")

    # ── Step 2: The Blind Council Vote ───────────────────────────────────────
    log.info(f"\n  {C.CYAN}[Step 2/4] The Blind Council Vote...{C.RESET}")

    X_holdout_scaled = mlp_scaler.transform(X_holdout)

    proba_lgbm = lgbm_model.predict_proba(X_holdout)
    proba_rf   = rf_model.predict_proba(X_holdout)
    proba_mlp  = mlp_model.predict_proba(X_holdout_scaled)
    ensemble_proba = (proba_lgbm + proba_rf + proba_mlp) / 3.0

    ref_classes = lgbm_model.classes_
    long_idx  = int(np.where(ref_classes == 1)[0][0])
    short_idx = int(np.where(ref_classes == -1)[0][0])
    chop_idx  = int(np.where(ref_classes == 0)[0][0])

    long_probs  = ensemble_proba[:, long_idx]
    short_probs = ensemble_proba[:, short_idx]

    log.info(f"    Ensemble proba matrix: {ensemble_proba.shape}")
    log.info(f"    Class map -> Long(1): col {long_idx}  |  Short(-1): col {short_idx}")

    # Apply thresholds
    long_mask  = long_probs >= lt
    short_mask = short_probs >= st
    both = long_mask & short_mask
    long_signal  = long_mask.copy()
    short_signal = short_mask.copy()
    long_signal[both]  = long_probs[both] >= short_probs[both]
    short_signal[both] = short_probs[both] > long_probs[both]
    triggered = long_signal | short_signal

    direction = np.zeros(len(y_holdout), dtype=int)
    direction[long_signal]  = 1
    direction[short_signal] = -1

    # ── Pre-Veto Stats ───────────────────────────────────────────────────────
    t_idx = np.where(triggered)[0]
    pre_lw  = np.sum((direction[t_idx] == 1)  & (y_holdout[t_idx] == 1))
    pre_ll  = np.sum((direction[t_idx] == 1)  & (y_holdout[t_idx] != 1))
    pre_sw  = np.sum((direction[t_idx] == -1) & (y_holdout[t_idx] == -1))
    pre_sl  = np.sum((direction[t_idx] == -1) & (y_holdout[t_idx] != -1))
    pre_wins   = pre_lw + pre_sw
    pre_losses = pre_ll + pre_sl
    pre_trades = pre_wins + pre_losses
    pre_total_r = (pre_lw * LONG_WIN_R + pre_ll * LONG_LOSS_R +
                   pre_sw * SHORT_WIN_R + pre_sl * SHORT_LOSS_R)
    pre_avg_r = pre_total_r / pre_trades if pre_trades > 0 else 0
    pre_wr    = pre_wins / pre_trades * 100 if pre_trades > 0 else 0

    log.info(f"    Pre-Veto: {C.YELLOW}{pre_trades:,}{C.RESET} trades triggered")

    # ── Step 3: The Blind Veto ───────────────────────────────────────────────
    log.info(f"\n  {C.CYAN}[Step 3/4] The Blind Veto (Meta-Model)...{C.RESET}")

    X_triggered = X_holdout.iloc[t_idx].copy()
    X_triggered['ens_long_proba']  = ensemble_proba[t_idx, long_idx]
    X_triggered['ens_short_proba'] = ensemble_proba[t_idx, short_idx]
    X_triggered['ens_chop_proba']  = ensemble_proba[t_idx, chop_idx]
    X_triggered['ens_direction']   = direction[t_idx]

    meta_pred = meta_model.predict(X_triggered)
    approved = meta_pred == 1

    # ── Post-Veto Stats ──────────────────────────────────────────────────────
    approved_idx = t_idx[approved]
    post_lw  = np.sum((direction[approved_idx] == 1)  & (y_holdout[approved_idx] == 1))
    post_ll  = np.sum((direction[approved_idx] == 1)  & (y_holdout[approved_idx] != 1))
    post_sw  = np.sum((direction[approved_idx] == -1) & (y_holdout[approved_idx] == -1))
    post_sl  = np.sum((direction[approved_idx] == -1) & (y_holdout[approved_idx] != -1))
    post_wins   = post_lw + post_sw
    post_losses = post_ll + post_sl
    post_trades = post_wins + post_losses
    post_total_r = (post_lw * LONG_WIN_R + post_ll * LONG_LOSS_R +
                    post_sw * SHORT_WIN_R + post_sl * SHORT_LOSS_R)
    post_avg_r = post_total_r / post_trades if post_trades > 0 else 0
    post_wr    = post_wins / post_trades * 100 if post_trades > 0 else 0

    vetoed = pre_trades - post_trades
    log.info(f"    Meta-Model vetoed {C.RED}{vetoed:,}{C.RESET} trades  |  {C.GREEN}{post_trades:,}{C.RESET} survived")

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 4 — THE FINAL VERDICT                              ║
    # ╚═══════════════════════════════════════════════════════════╝
    log.info(f"\n  {C.CYAN}[Step 4/4] The Final Verdict{C.RESET}")
    log.info("")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}        THE VAULT — OUT-OF-SAMPLE RESULTS{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.DIM}  Dataset: {holdout_path.name} ({len(X_holdout):,} unseen bars){C.RESET}")
    log.info(f"  {THIN}")
    log.info(f"  {'Metric':<28} {'PRE-VETO':>12}   {'POST-VETO':>12}")
    log.info(f"  {THIN}")
    log.info(f"  {'Trades Triggered':<28} {C.YELLOW}{pre_trades:>12,}{C.RESET}   {C.GREEN}{post_trades:>12,}{C.RESET}")
    log.info(f"  {'Trades Vetoed':<28} {'—':>12}   {C.RED}{vetoed:>12,}{C.RESET}")
    log.info(f"  {THIN}")
    log.info(f"  {'Long Wins':<28} {pre_lw:>12,}   {post_lw:>12,}")
    log.info(f"  {'Long Losses':<28} {pre_ll:>12,}   {post_ll:>12,}")
    log.info(f"  {'Short Wins':<28} {pre_sw:>12,}   {post_sw:>12,}")
    log.info(f"  {'Short Losses':<28} {pre_sl:>12,}   {post_sl:>12,}")
    log.info(f"  {THIN}")
    log.info(f"  {'Total Wins':<28} {C.GREEN}{pre_wins:>12,}{C.RESET}   {C.GREEN}{post_wins:>12,}{C.RESET}")
    log.info(f"  {'Total Losses':<28} {C.RED}{pre_losses:>12,}{C.RESET}   {C.RED}{post_losses:>12,}{C.RESET}")
    log.info(f"  {'Win Rate':<28} {pre_wr:>11.2f}%   {post_wr:>11.2f}%")
    log.info(f"  {THIN}")
    log.info(f"  {'Total R':<28} {pre_total_r:>11.2f}R   {C.BOLD}{C.GREEN}{post_total_r:>11.2f}R{C.RESET}")
    log.info(f"  {'Avg R per Trade':<28} {pre_avg_r:>11.2f}R   {C.BOLD}{C.GREEN}{post_avg_r:>11.2f}R{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    # Delta
    r_delta  = post_total_r - pre_total_r
    wr_delta = post_wr - pre_wr
    ar_delta = post_avg_r - pre_avg_r
    cr = C.GREEN if r_delta >= 0 else C.RED
    cw = C.GREEN if wr_delta >= 0 else C.RED
    ca = C.GREEN if ar_delta >= 0 else C.RED

    log.info(f"\n  {C.BOLD}VETO IMPACT (Out-of-Sample):{C.RESET}")
    log.info(f"    Total R Delta       : {cr}{r_delta:+.2f}R{C.RESET}")
    log.info(f"    Win Rate Delta      : {cw}{wr_delta:+.2f}%{C.RESET}")
    log.info(f"    Avg R/Trade Delta   : {ca}{ar_delta:+.2f}R{C.RESET}")
    log.info(f"    Signals Removed     : {vetoed:,} ({vetoed/pre_trades*100:.1f}% of all signals)")

    # ── Financial Tearsheet (both sides) ────────────────────────────────────
    def _calc_ts(idxs):
        dirs = direction[idxs]; ys = y_holdout[idxs]
        pnl = []
        for d, yv in zip(dirs, ys):
            if d == 1 and yv == 1:     pnl.append(LONG_WIN_R * RISK_PER_TRADE)
            elif d == 1:               pnl.append(LONG_LOSS_R * RISK_PER_TRADE)
            elif d == -1 and yv == -1: pnl.append(SHORT_WIN_R * RISK_PER_TRADE)
            else:                      pnl.append(SHORT_LOSS_R * RISK_PER_TRADE)
        pnl = np.array(pnl)
        if len(pnl) == 0:
            return {"net": 0, "final": STARTING_BALANCE, "dd": 0, "pf": 0, "mcl": 0, "r2d": 0}
        eq = STARTING_BALANCE + np.cumsum(pnl)
        pk = np.maximum.accumulate(eq)
        dd = (eq - pk).min()
        net = eq[-1] - STARTING_BALANCE
        gp = pnl[pnl > 0].sum(); gl = abs(pnl[pnl < 0].sum())
        pf = gp / gl if gl > 0 else float('inf')
        mcl = 0; s = 0
        for p in pnl:
            if p < 0: s += 1; mcl = max(mcl, s)
            else: s = 0
        r2d = abs(net / dd) if dd != 0 else float('inf')
        return {"net": net, "final": eq[-1], "dd": dd, "pf": pf, "mcl": mcl, "r2d": r2d}

    pre  = _calc_ts(t_idx)
    post = _calc_ts(approved_idx)

    log.info(f"\n  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}      FINANCIAL TEARSHEET  ($50 Risk / Trade){C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")
    log.info(f"  {THIN}")
    log.info(f"  {'Metric':<28} {'PRE-VETO':>14}   {'POST-VETO':>14}")
    log.info(f"  {THIN}")

    pc = C.GREEN if pre['net'] >= 0 else C.RED
    vc = C.GREEN if post['net'] >= 0 else C.RED
    log.info(f"  {'Net Profit':<28} {pc}${pre['net']:>13,.2f}{C.RESET}   {vc}${post['net']:>13,.2f}{C.RESET}")
    log.info(f"  {'Final Balance':<28} {pc}${pre['final']:>13,.2f}{C.RESET}   {vc}${post['final']:>13,.2f}{C.RESET}")
    log.info(f"  {'Max Drawdown':<28} {C.RED}${pre['dd']:>13,.2f}{C.RESET}   {C.RED}${post['dd']:>13,.2f}{C.RESET}")
    log.info(f"  {THIN}")

    ppf = C.GREEN if pre['pf'] > 1 else C.RED
    vpf = C.GREEN if post['pf'] > 1 else C.RED
    log.info(f"  {'Profit Factor':<28} {ppf}{pre['pf']:>15.2f}{C.RESET}   {vpf}{post['pf']:>15.2f}{C.RESET}")
    log.info(f"  {'Max Consec. Losses':<28} {C.RED}{pre['mcl']:>15}{C.RESET}   {C.RED}{post['mcl']:>15}{C.RESET}")

    pr2d = C.GREEN if pre['r2d'] > 1 else C.RED
    vr2d = C.GREEN if post['r2d'] > 1 else C.RED
    log.info(f"  {THIN}")
    log.info(f"  {'Reward/Drawdown Ratio':<28} {pr2d}{pre['r2d']:>15.2f}{C.RESET}   {vr2d}{post['r2d']:>15.2f}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIV}{C.RESET}")

    # Final Judgment
    log.info("")
    if post['net'] > 0 and post['pf'] > 1:
        log.info(f"  {C.BOLD}{C.GREEN}  >> THE VAULT IS PROFITABLE ON UNSEEN DATA <<{C.RESET}")
    elif post['net'] > 0:
        log.info(f"  {C.YELLOW}  >> Marginally profitable -- review trade quality.{C.RESET}")
    else:
        log.info(f"  {C.RED}  >> NEGATIVE EXPECTANCY -- Strategy does not generalize.{C.RESET}")
    log.info("")

if __name__ == "__main__":
    main()

