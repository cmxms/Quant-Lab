"""
ensemble_trainer.py
===================
The Brain: Soft-Voting Ensemble, Threshold Optimizer, & Meta-Model

Pipeline Stage : Model Training (Phase 3 - Final)
Inputs         : nq_final_clean_val.parquet  (from Feature Engineering/Outputs/)
                 lgbm_model_*.joblib, rf_model_*.joblib, mlp_model_*.joblib,
                 mlp_scaler_*.joblib  (from Model Training/Outputs/)
Outputs        : meta_model_YYYYMMDD_HHMMSS.joblib
                 optimal_thresholds_YYYYMMDD_HHMMSS.joblib

Usage
-----
    python ensemble_trainer.py
    python ensemble_trainer.py --prefix day_shift
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# ── ANSI Colors ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

class C:
    CYAN    = '\033[96m'
    GREEN   = '\033[92m'
    YELLOW  = '\033[93m'
    RED     = '\033[91m'
    MAGENTA = '\033[95m'
    WHITE   = '\033[97m'
    BOLD    = '\033[1m'
    RESET   = '\033[0m'

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE              = Path(__file__).parent.resolve()
_FEATURE_DIR       = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_DIR         = _HERE / "Outputs"

TARGET_COL: str    = "Target"
DROP_COLS: list    = [
    "Open", "High", "Low", "Close", "Volume",
    "Minutes_From_Open", "Day_Of_Week", "datetime", "ts_event",
]

# Phase 1 Barrier Parameters (R-Multiples: TP_mult / SL_mult)
LONG_WIN_R   =  2.44 / 0.53   # +4.60R per long win
LONG_LOSS_R  = -1.0            # -1.00R per long loss (1 unit of risk)
SHORT_WIN_R  =  1.93 / 0.54   # +3.57R per short win
SHORT_LOSS_R = -1.0            # -1.00R per short loss (1 unit of risk)

# Threshold grid
THRESH_LOW  = 0.40
THRESH_HIGH = 0.75
THRESH_STEP = 0.01
MIN_TRADES  = 1

# Meta-Model config
META_PARAMS: dict = {
    "objective":      "binary",
    "learning_rate":  0.05,
    "num_leaves":     31,
    "n_estimators":   500,
    "n_jobs":         -1,
    "random_state":   42,
    "class_weight":   "balanced",
}

# ---------------------------------------------------------------------------
# ── Logging ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

import os; os.system('')   # enable ANSI on Windows

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DIVIDER = "=" * 62

def _banner(title: str) -> None:
    log.info("")
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)

# ---------------------------------------------------------------------------
# ── Utilities ───────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _discover_latest(directory: Path, pattern: str) -> Path:
    """Return the most recently modified file matching a glob pattern."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"  {C.RED}[MISSING] No file matching '{pattern}' in {directory}{C.RESET}")
        sys.exit(1)
    chosen = matches[-1]
    log.info(f"  {C.GREEN}Found{C.RESET}: {chosen.name}")
    return chosen


def _load_val_data(val_substr: str = "val") -> tuple[pd.DataFrame, pd.Series]:
    """Load the validation parquet and split into X, y."""
    val_path = _discover_latest(_FEATURE_DIR, f"*{val_substr}*")
    df = pd.read_parquet(val_path)
    log.info(f"  Loaded validation set: {df.shape[0]:,} rows x {df.shape[1]} cols")

    # Find the target column (case-insensitive)
    target = TARGET_COL if TARGET_COL in df.columns else TARGET_COL.lower()
    if target not in df.columns:
        log.error(f"  {C.RED}Target column '{TARGET_COL}' not found.{C.RESET}")
        sys.exit(1)

    y = df[target]
    X = df.drop(columns=[target])

    # Drop non-feature columns
    to_drop = [c for c in DROP_COLS if c in X.columns]
    if to_drop:
        X = X.drop(columns=to_drop)
        log.info(f"  Dropped columns: {to_drop}")

    # Drop any remaining non-numeric columns
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        X = X.drop(columns=non_numeric)
        log.info(f"  Dropped non-numeric: {non_numeric}")

    log.info(f"  Features: {X.shape[1]}  |  Target distribution: {dict(y.value_counts().sort_index())}")
    return X, y

# ---------------------------------------------------------------------------
# ── Main Pipeline ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:

    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description="Ensemble Trainer — The Brain")
    parser.add_argument("--prefix", type=str, default="",
                        help="Regime prefix for model/data discovery and output naming (e.g. 'day_shift').")
    args = parser.parse_args()

    prefix = args.prefix
    model_prefix = f"{prefix}_" if prefix else ""
    val_substr   = f"{prefix}_clean_val" if prefix else "val"

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 1 — Model & Data Ingestion                         ║
    # ╚═══════════════════════════════════════════════════════════╝
    _banner("STEP 1 / 5  —  Model & Data Ingestion")

    log.info(f"\n  {C.CYAN}Loading base models (prefix='{model_prefix}')...{C.RESET}")
    lgbm_model  = joblib.load(_discover_latest(_MODEL_DIR, f"{model_prefix}lgbm_model_*.joblib"))
    rf_model    = joblib.load(_discover_latest(_MODEL_DIR, f"{model_prefix}rf_model_*.joblib"))
    mlp_model   = joblib.load(_discover_latest(_MODEL_DIR, f"{model_prefix}mlp_model_*.joblib"))
    mlp_scaler  = joblib.load(_discover_latest(_MODEL_DIR, f"{model_prefix}mlp_scaler_*.joblib"))

    log.info(f"\n  {C.CYAN}Loading validation data (substr='{val_substr}')...{C.RESET}")
    X_val, y_val = _load_val_data(val_substr)

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 2 — The Council  (Soft Voting)                     ║
    # ╚═══════════════════════════════════════════════════════════╝
    _banner("STEP 2 / 5  —  The Council  (Soft Voting)")

    # Scale data for the MLP only
    X_val_scaled = mlp_scaler.transform(X_val)

    # Generate probability matrices
    log.info(f"  {C.CYAN}Generating probability matrices...{C.RESET}")
    t0 = time.perf_counter()
    proba_lgbm = lgbm_model.predict_proba(X_val)
    proba_rf   = rf_model.predict_proba(X_val)
    proba_mlp  = mlp_model.predict_proba(X_val_scaled)
    elapsed = time.perf_counter() - t0

    log.info(f"  LGBM  proba shape: {proba_lgbm.shape}  |  classes: {lgbm_model.classes_}")
    log.info(f"  RF    proba shape: {proba_rf.shape}  |  classes: {rf_model.classes_}")
    log.info(f"  MLP   proba shape: {proba_mlp.shape}  |  classes: {mlp_model.classes_}")
    log.info(f"  Inference time: {elapsed:.2f}s")

    # Dynamic class index mapping
    # All models should have the same classes_ but we verify
    ref_classes = lgbm_model.classes_
    assert np.array_equal(ref_classes, rf_model.classes_), "RF classes mismatch!"
    assert np.array_equal(ref_classes, mlp_model.classes_), "MLP classes mismatch!"

    long_idx  = int(np.where(ref_classes == 1)[0][0])
    short_idx = int(np.where(ref_classes == -1)[0][0])
    chop_idx  = int(np.where(ref_classes == 0)[0][0])

    log.info(f"  Class mapping  ->  Long(1): col {long_idx}  |  Short(-1): col {short_idx}  |  Chop(0): col {chop_idx}")

    # Average the three probability matrices
    ensemble_proba = (proba_lgbm + proba_rf + proba_mlp) / 3.0
    log.info(f"  {C.GREEN}Ensemble probability matrix created: {ensemble_proba.shape}{C.RESET}")

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 3 — The Threshold Optimizer                        ║
    # ╚═══════════════════════════════════════════════════════════╝
    _banner("STEP 3 / 5  —  Threshold Optimizer  (Total R-Expectancy)")

    long_probs  = ensemble_proba[:, long_idx]
    short_probs = ensemble_proba[:, short_idx]
    y_arr       = y_val.to_numpy()

    thresholds = np.round(np.arange(THRESH_LOW, THRESH_HIGH + THRESH_STEP / 2, THRESH_STEP), 4)
    n_combos   = len(thresholds) ** 2
    log.info(f"  Grid: {len(thresholds)} thresholds x 2 directions = {n_combos:,} combinations")
    log.info(f"  R-Multiple parameters:")
    log.info(f"    Long  Win: +{LONG_WIN_R:.2f}R   |   Long  Loss: {LONG_LOSS_R:.2f}R")
    log.info(f"    Short Win: +{SHORT_WIN_R:.2f}R   |   Short Loss: {SHORT_LOSS_R:.2f}R")
    log.info(f"  Minimum trade count: {MIN_TRADES}")

    best_total_r = -np.inf
    best_long_thresh  = 0.0
    best_short_thresh = 0.0
    best_n_trades     = 0
    best_n_wins       = 0
    best_n_losses     = 0
    combos_evaluated  = 0

    t0 = time.perf_counter()

    for lt in thresholds:
        long_mask = long_probs >= lt

        for st in thresholds:
            short_mask = short_probs >= st

            # Resolve collisions: if both fire, pick the stronger signal
            both_fire = long_mask & short_mask
            long_signal  = long_mask.copy()
            short_signal = short_mask.copy()
            long_signal[both_fire]  = long_probs[both_fire] >= short_probs[both_fire]
            short_signal[both_fire] = short_probs[both_fire] > long_probs[both_fire]

            # Calculate expectancy
            # Long trades
            long_wins  = np.sum(long_signal & (y_arr == 1))
            long_losses = np.sum(long_signal & (y_arr != 1))

            # Short trades
            short_wins  = np.sum(short_signal & (y_arr == -1))
            short_losses = np.sum(short_signal & (y_arr != -1))

            n_trades = long_wins + long_losses + short_wins + short_losses

            if n_trades < MIN_TRADES:
                continue

            total_r = (
                long_wins    * LONG_WIN_R  +
                long_losses  * LONG_LOSS_R +
                short_wins   * SHORT_WIN_R +
                short_losses * SHORT_LOSS_R
            )

            combos_evaluated += 1

            if total_r > best_total_r:
                best_total_r      = total_r
                best_long_thresh  = lt
                best_short_thresh = st
                best_n_trades     = n_trades
                best_n_wins       = long_wins + short_wins
                best_n_losses     = long_losses + short_losses

    elapsed = time.perf_counter() - t0
    log.info(f"  Evaluated {combos_evaluated:,} valid combinations in {elapsed:.2f}s")

    if best_total_r == -np.inf:
        log.error(f"  {C.RED}No threshold combination met the {MIN_TRADES}-trade minimum!{C.RESET}")
        sys.exit(1)

    base_win_rate = best_n_wins / best_n_trades * 100 if best_n_trades > 0 else 0

    log.info("")
    log.info(f"  {C.BOLD}{C.GREEN}>>> OPTIMAL THRESHOLDS FOUND <<<{C.RESET}")
    log.info(f"  {C.CYAN}Long  Threshold : {best_long_thresh:.2f}{C.RESET}")
    log.info(f"  {C.CYAN}Short Threshold : {best_short_thresh:.2f}{C.RESET}")
    log.info(f"  {C.GREEN}Total R Gained  : {best_total_r:,.2f}R{C.RESET}")
    avg_r = best_total_r / best_n_trades if best_n_trades > 0 else 0
    log.info(f"  {C.GREEN}Avg R per Trade : {avg_r:,.2f}R{C.RESET}")
    log.info(f"  {C.WHITE}Trades Triggered: {best_n_trades:,}  (W: {best_n_wins:,} / L: {best_n_losses:,}){C.RESET}")
    log.info(f"  {C.WHITE}Council Win Rate: {base_win_rate:.2f}%{C.RESET}")

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 4 — The Meta-Model  (The Veto Layer)               ║
    # ╚═══════════════════════════════════════════════════════════╝
    _banner("STEP 4 / 5  —  The Meta-Model  (Veto Layer)")

    # Reconstruct the optimal signal masks
    lt_mask = long_probs >= best_long_thresh
    st_mask = short_probs >= best_short_thresh

    both = lt_mask & st_mask
    final_long  = lt_mask.copy()
    final_short = st_mask.copy()
    final_long[both]  = long_probs[both] >= short_probs[both]
    final_short[both] = short_probs[both] > long_probs[both]

    triggered = final_long | final_short

    # Determine ensemble direction for triggered rows
    ensemble_direction = np.zeros(len(y_arr), dtype=int)
    ensemble_direction[final_long]  = 1
    ensemble_direction[final_short] = -1

    # Filter down to triggered rows
    X_triggered = X_val.loc[triggered].copy()
    y_triggered = y_arr[triggered]
    dir_triggered = ensemble_direction[triggered]

    # Create Is_Correct binary target
    is_correct = (dir_triggered == y_triggered).astype(int)

    log.info(f"  Triggered rows: {len(X_triggered):,}")
    log.info(f"  Is_Correct distribution: {{1: {np.sum(is_correct)}, 0: {len(is_correct) - np.sum(is_correct)}}}")

    # Append ensemble probabilities as meta-features
    ens_proba_triggered = ensemble_proba[triggered]
    meta_features = X_triggered.copy()
    meta_features['ens_long_proba']  = ens_proba_triggered[:, long_idx]
    meta_features['ens_short_proba'] = ens_proba_triggered[:, short_idx]
    meta_features['ens_chop_proba']  = ens_proba_triggered[:, chop_idx]
    meta_features['ens_direction']   = dir_triggered

    # Internal train/val split for early stopping
    X_meta_train, X_meta_val, y_meta_train, y_meta_val = train_test_split(
        meta_features, is_correct, test_size=0.2, random_state=42, stratify=is_correct
    )

    log.info(f"  Meta-Train: {len(X_meta_train):,}  |  Meta-Val: {len(X_meta_val):,}")
    log.info(f"  {C.CYAN}Training Meta-Model (LGBMClassifier)...{C.RESET}")

    meta_model = lgb.LGBMClassifier(**META_PARAMS)
    meta_model.fit(
        X_meta_train, y_meta_train,
        eval_set=[(X_meta_val, y_meta_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )

    # Quick meta-model evaluation
    meta_pred = meta_model.predict(X_meta_val)
    meta_acc = accuracy_score(y_meta_val, meta_pred)
    log.info(f"  {C.GREEN}Meta-Model Accuracy (internal val): {meta_acc:.4f}  ({meta_acc * 100:.2f}%){C.RESET}")
    log.info(f"  Best iteration: {meta_model.best_iteration_}")

    # ╔═══════════════════════════════════════════════════════════╗
    # ║  STEP 5 — Serialise & Report                             ║
    # ╚═══════════════════════════════════════════════════════════╝
    _banner("STEP 5 / 5  —  Serialise & Final Report")

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Meta-Model
    meta_path = _MODEL_DIR / f"{model_prefix}meta_model_{timestamp}.joblib"
    joblib.dump(meta_model, meta_path)
    log.info(f"  {C.GREEN}Meta-Model saved -> {meta_path.name}  ({meta_path.stat().st_size / 1_048_576:.2f} MB){C.RESET}")

    # Save optimal thresholds
    threshold_dict = {
        "long_threshold":   best_long_thresh,
        "short_threshold":  best_short_thresh,
        "total_r":          best_total_r,
        "avg_r_per_trade":  avg_r,
        "n_trades":         best_n_trades,
        "n_wins":           best_n_wins,
        "n_losses":         best_n_losses,
        "base_win_rate":    base_win_rate,
        "long_win_r":       LONG_WIN_R,
        "long_loss_r":      LONG_LOSS_R,
        "short_win_r":      SHORT_WIN_R,
        "short_loss_r":     SHORT_LOSS_R,
        "meta_model_acc":   meta_acc,
        "timestamp":        timestamp,
    }
    thresh_path = _MODEL_DIR / f"{model_prefix}optimal_thresholds_{timestamp}.joblib"
    joblib.dump(threshold_dict, thresh_path)
    log.info(f"  {C.GREEN}Thresholds saved -> {thresh_path.name}{C.RESET}")

    # ── Final Console Report ─────────────────────────────────────────────────
    log.info("")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIVIDER}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}            THE BRAIN — FINAL REPORT{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIVIDER}{C.RESET}")
    log.info(f"  {C.WHITE}Optimal Long  Threshold  : {C.CYAN}{best_long_thresh:.2f}{C.RESET}")
    log.info(f"  {C.WHITE}Optimal Short Threshold  : {C.CYAN}{best_short_thresh:.2f}{C.RESET}")
    log.info(f"  {C.WHITE}Council Base Win Rate     : {C.GREEN}{base_win_rate:.2f}%{C.RESET}")
    log.info(f"  {C.WHITE}Total Trades Triggered    : {C.YELLOW}{best_n_trades:,}{C.RESET}")
    log.info(f"  {C.WHITE}  -> Wins                 : {C.GREEN}{best_n_wins:,}{C.RESET}")
    log.info(f"  {C.WHITE}  -> Losses               : {C.RED}{best_n_losses:,}{C.RESET}")
    log.info(f"  {C.WHITE}Total R Gained            : {C.BOLD}{C.GREEN}{best_total_r:,.2f}R{C.RESET}")
    log.info(f"  {C.WHITE}Avg R per Trade           : {C.BOLD}{C.GREEN}{avg_r:,.2f}R{C.RESET}")
    log.info(f"  {C.WHITE}Meta-Model Accuracy       : {C.GREEN}{meta_acc:.4f}{C.RESET}")
    log.info(f"  {C.BOLD}{C.MAGENTA}{DIVIDER}{C.RESET}")
    log.info("")


if __name__ == "__main__":
    main()
