"""
orchestrator.py
===============
Ensemble Orchestrator -- Soft Voting on Sealed Holdout Data

Pipeline Stage : Final Evaluation (Phase 4)
Inputs         : Feature Engineering/Outputs/long_clean_holdout.parquet  (SEALED)
                 Model Training/Outputs/rf_model_*.joblib
                 Model Training/Outputs/lgbm_model_*.joblib
                 Model Training/Outputs/logreg_pipeline_*.joblib
Outputs        : Console report (accuracy, confusion matrix, per-model comparison)

Strategy
--------
Soft Voting Ensemble: Each model produces class probabilities via predict_proba().
The three probability arrays are averaged into a single ensemble probability vector.
np.argmax selects the final prediction. This is the "live market" simulation --
the holdout set has never been seen by any model or the feature selection process.

CRITICAL -- Class Alignment
---------------------------
Each model's predict_proba() columns correspond to model.classes_ which may not
be in a consistent order across models. All probability arrays are explicitly
reordered to the canonical class order [-1, 0, 1] before averaging.

Usage
-----
    python orchestrator.py
    python orchestrator.py --holdout_path /path/to/holdout.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE               = Path(__file__).parent.resolve()
_FEATURE_OUTPUT_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_OUTPUT_DIR   = _HERE / "Outputs"

TARGET_COL: str  = "Target"
DROP_COLS: list  = [
    "Open", "High", "Low", "Close", "Volume",
    "Minutes_From_Open", "Day_Of_Week",
]

CANONICAL_CLASSES = np.array([-1, 0, 1])   # Enforced column order for all proba arrays

# ---------------------------------------------------------------------------
# ── Logging Setup ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DIVIDER = "-" * 60


def _section(title: str) -> None:
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)


# ---------------------------------------------------------------------------
# ── Helpers ─────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _find_latest(directory: Path, pattern: str, label: str) -> Path:
    """Return the most recently modified file matching pattern."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"[NO MODEL] No '{pattern}' found in {directory}")
        sys.exit(1)
    latest = matches[-1]
    log.info(f"  {label:<12}: {latest.name}  ({len(matches)} version(s) found)")
    return latest


def _align_proba(proba: np.ndarray, model_classes: np.ndarray) -> np.ndarray:
    """
    Reorder predict_proba() output columns to match CANONICAL_CLASSES [-1, 0, 1].

    Args:
        proba        : (n_samples, n_model_classes) array from predict_proba()
        model_classes: model.classes_ array (the ordering used during training)

    Returns:
        (n_samples, 3) array with columns in canonical [-1, 0, 1] order.
    """
    aligned = np.zeros((proba.shape[0], len(CANONICAL_CLASSES)), dtype=np.float64)
    for target_idx, cls in enumerate(CANONICAL_CLASSES):
        model_col = np.where(model_classes == cls)[0]
        if len(model_col) == 0:
            log.warning(f"  [ALIGN] Class {cls} not in model.classes_ -- filling with 0")
        else:
            aligned[:, target_idx] = proba[:, model_col[0]]
    return aligned


def _print_confusion_matrix(cm: np.ndarray, classes: np.ndarray) -> None:
    col_width = max(len(str(c)) for c in classes) + 4
    log.info("  Confusion Matrix  (rows = Actual, cols = Predicted)")
    log.info("")
    header = "  Actual v / Predicted -> " + "".join(
        str(c).center(col_width) for c in classes
    )
    log.info(header)
    log.info("  " + "-" * (len(header) - 2))
    for i, row in enumerate(cm):
        row_str = f"  Class {classes[i]:<5}" + "".join(
            str(v).center(col_width) for v in row
        )
        log.info(row_str)
    log.info("")


def _evaluate(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> float:
    """Compute and log accuracy, weighted precision/recall, per-class breakdown."""
    classes    = CANONICAL_CLASSES
    avg        = "weighted"
    accuracy   = accuracy_score(y_true, y_pred)
    precision  = precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall     = recall_score(y_true, y_pred, average=avg, zero_division=0)
    cm         = confusion_matrix(y_true, y_pred, labels=classes)

    log.info(f"  Accuracy          : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    log.info(f"  Precision (wt.)   : {precision:.4f}")
    log.info(f"  Recall    (wt.)   : {recall:.4f}")
    log.info("")
    _print_confusion_matrix(cm, classes)

    per_prec = precision_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    per_rec  = recall_score(y_true, y_pred, average=None, labels=classes, zero_division=0)
    log.info("  Per-class breakdown:")
    for cls, prec, rec in zip(classes, per_prec, per_rec):
        log.info(f"    Class {cls:<2} ->  Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    log.info("")
    return accuracy


# ---------------------------------------------------------------------------
# ── Main ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Soft Voting Ensemble Orchestrator -- evaluate on sealed holdout data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n  python orchestrator.py\n  python orchestrator.py --holdout_path /path/to/holdout.parquet",
    )
    parser.add_argument(
        "--holdout_path",
        type=Path,
        default=None,
        help="Override: explicit path to the holdout .parquet file.",
    )
    args = parser.parse_args()

    log.info("")
    _section("Ensemble Orchestrator  -  Quantitative Trading Pipeline  |  Phase 4")
    log.info("  Strategy : Soft Voting (averaged predict_proba across 3 models)")
    log.info("  Dataset  : SEALED holdout -- never seen by any model or MI selector")
    log.info("")

    # ── 1. Load Models ────────────────────────────────────────────────────────
    _section("Step 1 / 4  -  Loading Models")
    rf_path     = _find_latest(_MODEL_OUTPUT_DIR, "rf_model_*.joblib",       "RF")
    lgbm_path   = _find_latest(_MODEL_OUTPUT_DIR, "lgbm_model_*.joblib",     "LGBM")
    logreg_path = _find_latest(_MODEL_OUTPUT_DIR, "logreg_pipeline_*.joblib","LogReg")

    log.info("")
    log.info("  Deserialising models ...")
    rf_model     = joblib.load(rf_path)
    lgbm_model   = joblib.load(lgbm_path)
    logreg_pipe  = joblib.load(logreg_path)   # Full Pipeline: scaler + logreg

    # Extract inner LogReg for class inspection
    logreg_inner = logreg_pipe.named_steps["model"]

    log.info(f"  RF     classes : {rf_model.classes_}")
    log.info(f"  LGBM   classes : {lgbm_model.classes_}")
    log.info(f"  LogReg classes : {logreg_inner.classes_}")
    log.info("  Canonical order: [-1  0  1]")

    # ── 2. Load & Prepare Holdout Data ───────────────────────────────────────
    _section("Step 2 / 4  -  Loading Sealed Holdout Data")

    holdout_file = args.holdout_path or (_FEATURE_OUTPUT_DIR / "long_clean_holdout.parquet")
    if not holdout_file.exists():
        log.error(f"[MISSING FILE] Holdout file not found: {holdout_file}")
        sys.exit(1)

    log.info(f"  Reading: {holdout_file.name}")
    df_holdout = pd.read_parquet(holdout_file)
    log.info(f"  Full shape: {df_holdout.shape[0]:,} rows x {df_holdout.shape[1]} columns")

    if TARGET_COL not in df_holdout.columns:
        log.error(f"[MISSING TARGET] Column '{TARGET_COL}' not found in holdout data.")
        sys.exit(1)

    y_true = df_holdout[TARGET_COL].to_numpy()
    drop   = [c for c in [TARGET_COL] + DROP_COLS if c in df_holdout.columns]
    X      = df_holdout.drop(columns=drop)

    log.info(f"  Feature matrix  : {X.shape[0]:,} rows x {X.shape[1]} features")
    log.info(f"  Features        : {list(X.columns)}")
    log.info(f"  Ground truth    : {len(y_true):,} labels")
    log.info(f"  Class counts    : { dict(pd.Series(y_true).value_counts().sort_index()) }")

    # ── 3. Soft Vote Engine ───────────────────────────────────────────────────
    _section("Step 3 / 4  -  Soft Vote Engine")

    log.info("  Running predict_proba() on all three models ...")
    log.info("")

    # RF -- raw features
    log.info("  [1/3] Random Forest ...")
    rf_proba_raw = rf_model.predict_proba(X)
    rf_proba     = _align_proba(rf_proba_raw, rf_model.classes_)
    log.info(f"        Output shape : {rf_proba.shape}  (aligned to [-1, 0, 1])")

    # LGBM -- raw features
    log.info("  [2/3] LightGBM ...")
    lgbm_proba_raw = lgbm_model.predict_proba(X)
    lgbm_proba     = _align_proba(lgbm_proba_raw, lgbm_model.classes_)
    log.info(f"        Output shape : {lgbm_proba.shape}  (aligned to [-1, 0, 1])")

    # LogReg -- Pipeline handles scaling internally
    log.info("  [3/3] Logistic Regression (scaler applied internally by Pipeline) ...")
    logreg_proba_raw = logreg_pipe.predict_proba(X)
    logreg_proba     = _align_proba(logreg_proba_raw, logreg_inner.classes_)
    log.info(f"        Output shape : {logreg_proba.shape}  (aligned to [-1, 0, 1])")

    # Average probabilities -- soft vote
    log.info("")
    log.info("  Averaging probability arrays (equal weights: 1/3 each) ...")
    ensemble_proba = (rf_proba + lgbm_proba + logreg_proba) / 3.0
    log.info(f"  Ensemble proba shape: {ensemble_proba.shape}")

    # Decode argmax back to class labels
    argmax_idx     = np.argmax(ensemble_proba, axis=1)
    y_ensemble     = CANONICAL_CLASSES[argmax_idx]

    # Individual standalone predictions for comparison
    y_rf     = CANONICAL_CLASSES[np.argmax(rf_proba,     axis=1)]
    y_lgbm   = CANONICAL_CLASSES[np.argmax(lgbm_proba,   axis=1)]
    y_logreg = CANONICAL_CLASSES[np.argmax(logreg_proba, axis=1)]

    # ── 4. Evaluation Ledger ─────────────────────────────────────────────────
    _section("Step 4 / 4  -  The Ledger  (Sealed Holdout -- Never Before Seen)")

    # ── Ensemble ──────────────────────────────────────────────────────────────
    log.info(DIVIDER)
    log.info("  *** SOFT VOTING ENSEMBLE  (RF + LGBM + LogReg) ***")
    log.info(DIVIDER)
    ens_acc = _evaluate(y_true, y_ensemble, "Ensemble")

    # ── Individual Models ─────────────────────────────────────────────────────
    log.info(DIVIDER)
    log.info("  Individual Model Comparison  (same holdout data)")
    log.info(DIVIDER)
    log.info("")

    results = {}
    for label, y_pred in [("RF", y_rf), ("LGBM", y_lgbm), ("LogReg", y_logreg)]:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_true, y_pred,    average="weighted", zero_division=0)
        results[label] = (acc, prec, rec)

    # Print comparison table
    log.info(f"  {'Model':<12} {'Accuracy':>10} {'Precision':>11} {'Recall':>9}  Delta vs Ensemble")
    log.info(f"  {'-'*12} {'-'*10} {'-'*11} {'-'*9}  {'-'*20}")
    for label, (acc, prec, rec) in results.items():
        delta = ens_acc - acc
        direction = "^" if delta > 0 else ("v" if delta < 0 else "=")
        log.info(
            f"  {label:<12} {acc:>10.4f} {prec:>11.4f} {rec:>9.4f}"
            f"  {direction} {abs(delta)*100:+.2f}%"
        )
    log.info(f"  {'ENSEMBLE':<12} {ens_acc:>10.4f}  <- final")
    log.info("")

    # Verdict
    best_individual = max(results.items(), key=lambda x: x[1][0])
    log.info(DIVIDER)
    if ens_acc > best_individual[1][0]:
        margin = (ens_acc - best_individual[1][0]) * 100
        log.info(f"  VERDICT: Ensemble OUTPERFORMS best individual ({best_individual[0]}) "
                 f"by +{margin:.2f}%")
    elif ens_acc == best_individual[1][0]:
        log.info(f"  VERDICT: Ensemble MATCHES best individual ({best_individual[0]})")
    else:
        margin = (best_individual[1][0] - ens_acc) * 100
        log.info(f"  VERDICT: Best individual ({best_individual[0]}) outperforms "
                 f"ensemble by +{margin:.2f}%")
    log.info(DIVIDER)
    log.info("")
    _section("Done  [v]  orchestrator.py completed successfully")
    log.info("")


if __name__ == "__main__":
    main()
