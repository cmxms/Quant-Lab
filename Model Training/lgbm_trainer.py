"""
lgbm_trainer.py
===============
Standalone LightGBM Classifier -- Train & Validate

Pipeline Stage : Model Training (Phase 3)
Inputs         : Any .parquet file containing 'train' / 'val' in its name
                 Auto-discovered in ../Feature Engineering/Outputs/
Outputs        : Outputs/lgbm_model_YYYYMMDD_HHMMSS.joblib

Usage
-----
    # Auto-discover files by substring match:
    python lgbm_trainer.py

    # Override one or both paths explicitly:
    python lgbm_trainer.py --train_path /path/to/train.parquet
    python lgbm_trainer.py --train_path /path/to/train.parquet --val_path /path/to/val.parquet
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
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
_FEATURE_OUTPUT_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_OUTPUT_DIR = _HERE / "Outputs"

TARGET_COL: str = "Target"
DROP_COLS: list = [
    "Open", "High", "Low", "Close", "Volume",
    "Minutes_From_Open", "Day_Of_Week",
]

LGBM_PARAMS: dict = {
    "objective": "multiclass",
    "class_weight": "balanced",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "n_estimators": 2000,
    "n_jobs": -1,
    "random_state": 42,
}

EARLY_STOPPING_ROUNDS: int = 100

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

# ---------------------------------------------------------------------------
# ── Helper Utilities ────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

DIVIDER = "-" * 60


def _section(title: str) -> None:
    log.info(DIVIDER)
    log.info(f"  {title}")
    log.info(DIVIDER)


def _discover_file(directory: Path, substring: str) -> Path:
    """Scan directory for the first .parquet file whose name contains substring.

    Args:
        directory: Directory to scan.
        substring: Case-insensitive substring to match against filenames.

    Returns:
        Resolved Path to the matched file.

    Exits:
        With code 1 if no match is found or the directory does not exist.
    """
    if not directory.exists():
        log.error(f"[MISSING DIR] Feature output directory not found: {directory}")
        sys.exit(1)

    matches = [
        p for p in sorted(directory.glob("*.parquet"))
        if substring.lower() in p.stem.lower()
    ]

    if not matches:
        log.error(
            f"[NO MATCH] No .parquet file containing '{substring}' found in: {directory}\n"
            f"  Files present: {[p.name for p in directory.glob('*.parquet')]}"
        )
        sys.exit(1)

    if len(matches) > 1:
        log.warning(
            f"  [AMBIGUOUS] Multiple files match '{substring}': "
            f"{[m.name for m in matches]}. Using first: '{matches[0].name}'"
        )

    return matches[0]


def _load_parquet(path: Path, label: str) -> pd.DataFrame:
    if not path.exists():
        log.error(f"[MISSING FILE] {label} file not found at: {path}")
        sys.exit(1)
    log.info(f"  Reading {label}: {path.name}")
    df = pd.read_parquet(path)
    log.info(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


def _split_xy(df: pd.DataFrame, target: str, dataset_label: str):
    """Extract target vector, drop the target column and non-stationary features."""
    if target not in df.columns:
        log.error(f"[TARGET NOT FOUND] Column '{target}' missing in {dataset_label} dataset.")
        sys.exit(1)

    y = df[target]
    X = df.drop(columns=[target])

    # Drop non-stationary / time-based features
    cols_to_drop = [c for c in DROP_COLS if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        log.info(f"  Dropped non-stationary columns: {cols_to_drop}")

    log.info(f"  Final feature set ({X.shape[1]} features): {list(X.columns)}")
    log.info(f"  {dataset_label} -> features: {X.shape[1]}  |  target: '{target}'")
    return X, y


def _print_confusion_matrix(cm: np.ndarray, classes: np.ndarray) -> None:
    col_width = max(len(str(c)) for c in classes) + 4

    log.info("  Confusion Matrix  (rows = Actual, cols = Predicted)")
    log.info("")
    header_row = "  Actual v / Predicted -> " + "".join(
        str(c).center(col_width) for c in classes
    )
    log.info(header_row)
    log.info("  " + "-" * (len(header_row) - 2))
    for i, row in enumerate(cm):
        row_str = f"  Class {classes[i]:<5}" + "".join(
            str(v).center(col_width) for v in row
        )
        log.info(row_str)
    log.info("")


# ---------------------------------------------------------------------------
# ── Main Pipeline ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    # ── CLI Argument Parsing ──────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Train and validate a LightGBM classifier for the Quant Lab pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python lgbm_trainer.py\n"
            "  python lgbm_trainer.py --train_path /path/to/train.parquet\n"
            "  python lgbm_trainer.py --train_path /path/to/train.parquet --val_path /path/to/val.parquet"
        ),
    )
    parser.add_argument(
        "--train_path",
        type=Path,
        default=None,
        help="Override: explicit path to the training .parquet file.",
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=None,
        help="Override: explicit path to the validation .parquet file.",
    )
    args = parser.parse_args()

    log.info("")
    _section("LGBM Trainer  -  Quantitative Trading Pipeline  |  Phase 3")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    _section("Step 1 / 4  -  Loading Data")

    # Resolve train file: CLI override takes priority, else auto-discover
    if args.train_path:
        train_file = args.train_path
        log.info(f"  Train file  : [CLI override] {train_file}")
    else:
        log.info(f"  Scanning for 'train' parquet in: {_FEATURE_OUTPUT_DIR}")
        train_file = _discover_file(_FEATURE_OUTPUT_DIR, "train")
        log.info(f"  Train file  : [auto-discovered] {train_file.name}")

    # Resolve val file: CLI override takes priority, else auto-discover
    if args.val_path:
        val_file = args.val_path
        log.info(f"  Val file    : [CLI override] {val_file}")
    else:
        log.info(f"  Scanning for 'val' parquet in: {_FEATURE_OUTPUT_DIR}")
        val_file = _discover_file(_FEATURE_OUTPUT_DIR, "val")
        log.info(f"  Val file    : [auto-discovered] {val_file.name}")

    df_train = _load_parquet(train_file, "Train")
    df_val   = _load_parquet(val_file, "Validation")

    X_train, y_train = _split_xy(df_train, TARGET_COL, "Train")
    X_val,   y_val   = _split_xy(df_val,   TARGET_COL, "Validation")

    # Verify feature columns match between splits
    missing_in_val = set(X_train.columns) - set(X_val.columns)
    if missing_in_val:
        log.warning(f"  [COLUMN MISMATCH] Features in train but not val: {missing_in_val}")
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    log.info(f"  Class distribution (train): {dict(y_train.value_counts().sort_index())}")
    log.info(f"  Class distribution (val)  : {dict(y_val.value_counts().sort_index())}")

    # ── 2. Initialise Model ───────────────────────────────────────────────────
    _section("Step 2 / 4  -  Initialising LightGBM Classifier")

    # Derive num_class from training data for multiclass logging
    n_classes = y_train.nunique()
    log.info(f"  Detected classes          : {sorted(y_train.unique())}  (n={n_classes})")
    log.info("")
    for k, v in LGBM_PARAMS.items():
        log.info(f"  {k:<18}: {v}")
    log.info(f"  {'early_stopping':<18}: {EARLY_STOPPING_ROUNDS} rounds")
    log.info("")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    # ── 3. Train with Early Stopping ─────────────────────────────────────────
    _section("Step 3 / 4  -  Training Model (Early Stopping Active)")
    log.info(
        f"  Fitting on {X_train.shape[0]:,} training samples "
        f"with {X_train.shape[1]} features ..."
    )
    log.info(f"  Validating on {X_val.shape[0]:,} samples  |  ES patience: {EARLY_STOPPING_ROUNDS}")
    log.info("")

    callbacks = [
        lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
        lgb.log_evaluation(period=100),  # print eval metric every 100 rounds
    ]

    t0 = time.perf_counter()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks,
    )
    elapsed = time.perf_counter() - t0

    best_iter = model.best_iteration_
    log.info("")
    log.info(f"  Training complete in {elapsed:.1f}s")
    log.info(f"  Best iteration            : {best_iter}")

    # ── 4. Evaluate on Validation Set ────────────────────────────────────────
    _section("Step 4 / 4  -  Evaluation Results  (Validation Set)")
    y_pred = model.predict(X_val)

    classes    = np.sort(np.unique(y_val))
    avg_method = "weighted"

    accuracy  = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average=avg_method, zero_division=0)
    recall    = recall_score(y_val, y_pred, average=avg_method, zero_division=0)
    cm        = confusion_matrix(y_val, y_pred, labels=classes)

    log.info(f"  Accuracy          : {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    log.info(f"  Precision (wt.)   : {precision:.4f}")
    log.info(f"  Recall    (wt.)   : {recall:.4f}")
    log.info("")
    _print_confusion_matrix(cm, classes)

    per_class_prec = precision_score(y_val, y_pred, average=None, labels=classes, zero_division=0)
    per_class_rec  = recall_score(y_val, y_pred, average=None, labels=classes, zero_division=0)
    log.info("  Per-class breakdown:")
    for cls, prec, rec in zip(classes, per_class_prec, per_class_rec):
        log.info(f"    Class {cls:<2} ->  Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    log.info("")

    # Feature importance (gain-based, consistent across tree methods)
    importances = pd.Series(
        model.booster_.feature_importance(importance_type="gain"),
        index=X_train.columns,
    )
    top10 = importances.sort_values(ascending=False).head(10)
    log.info("  Top-10 Feature Importances (Gain):")
    max_imp = top10.iloc[0] if top10.iloc[0] > 0 else 1.0
    for feat, imp in top10.items():
        bar = "=" * int((imp / max_imp) * 40)
        log.info(f"    {feat:<25} {imp:>10.2f}  {bar}")
    log.info("")

    # ── 5. Serialise Model ────────────────────────────────────────────────────
    _section("Serialising Model")
    _MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"lgbm_model_{timestamp}.joblib"
    model_path     = _MODEL_OUTPUT_DIR / model_filename

    joblib.dump(model, model_path)
    size_mb = model_path.stat().st_size / 1_048_576
    log.info(f"  Model saved -> {model_path}")
    log.info(f"  File size   : {size_mb:.2f} MB")
    log.info("")
    _section("Done  [v]  lgbm_trainer.py completed successfully")
    log.info("")


if __name__ == "__main__":
    main()
