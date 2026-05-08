"""
mlp_trainer.py
=============
Standalone Multi-Layer Perceptron (MLP) Classifier -- Train & Validate

Pipeline Stage : Model Training (Phase 3)
Inputs         : Any .parquet file containing 'train' / 'val' in its name
                 Auto-discovered in ../Feature Engineering/Outputs/
Outputs        : Outputs/mlp_model_YYYYMMDD_HHMMSS.joblib
                 Outputs/mlp_scaler_YYYYMMDD_HHMMSS.joblib

Usage
-----
    # Auto-discover files by substring match:
    python mlp_trainer.py

    # Override one or both paths explicitly:
    python mlp_trainer.py --train_path /path/to/train.parquet
    python mlp_trainer.py --train_path /path/to/train.parquet --val_path /path/to/val.parquet
"""

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
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
DROP_COLS: list = ["Open", "High", "Low", "Close", "Volume", "Minutes_From_Open", "Day_Of_Week", "datetime", "ts_event"]

MLP_PARAMS: dict = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation": "relu",
    "solver": "adam",
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "random_state": 42,
    "verbose": True,
}

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
    """Scan directory for the first .parquet file whose name contains substring."""
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
    if target not in df.columns:
        # Check for lowercase 'target' fallback
        if target.lower() in df.columns:
            target = target.lower()
        else:
            log.error(f"[TARGET NOT FOUND] Column '{target}' missing in {dataset_label} dataset.")
            sys.exit(1)

    y = df[target]
    X = df.drop(columns=[target])
    
    # Drop non-stationary features
    cols_to_drop = [c for c in DROP_COLS if c in X.columns]
    if cols_to_drop:
        X = X.drop(columns=cols_to_drop)
        log.info(f"  Dropped non-stationary columns: {cols_to_drop}")

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
        description="Train and validate an MLP classifier for the Quant Lab pipeline.",
    )
    parser.add_argument("--train_path", type=Path, default=None)
    parser.add_argument("--val_path", type=Path, default=None)
    parser.add_argument("--prefix", type=str, default="",
                        help="Regime prefix for file discovery and output naming (e.g. 'day_shift').")
    args = parser.parse_args()

    prefix = args.prefix
    train_substr = f"{prefix}_clean_train" if prefix else "train"
    val_substr   = f"{prefix}_clean_val" if prefix else "val"
    file_prefix  = f"{prefix}_" if prefix else ""

    log.info("")
    _section("MLP Trainer  -  Quantitative Trading Pipeline  |  Phase 3")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    _section("Step 1 / 5  -  Loading Data")

    if args.train_path:
        train_file = args.train_path
    else:
        train_file = _discover_file(_FEATURE_OUTPUT_DIR, train_substr)

    if args.val_path:
        val_file = args.val_path
    else:
        val_file = _discover_file(_FEATURE_OUTPUT_DIR, val_substr)

    df_train = _load_parquet(train_file, "Train")
    df_val   = _load_parquet(val_file, "Validation")

    X_train, y_train = _split_xy(df_train, TARGET_COL, "Train")
    X_val, y_val = _split_xy(df_val, TARGET_COL, "Validation")

    log.info(f"  Class distribution (train): {dict(y_train.value_counts().sort_index())}")
    log.info(f"  Class distribution (val)  : {dict(y_val.value_counts().sort_index())}")

    # ── 2. Standardization ────────────────────────────────────────────────────
    _section("Step 2 / 5  -  Standardizing Features")
    log.info("  Fitting StandardScaler on training data to prevent data leakage.")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    log.info("  Applying scaler to validation data.")

    # ── 3. Initialise Model ───────────────────────────────────────────────────
    _section("Step 3 / 5  -  Initialising Neural Network (MLP)")
    for k, v in MLP_PARAMS.items():
        log.info(f"  {k:<18}: {v}")

    model = MLPClassifier(**MLP_PARAMS)

    # ── 4. Train ──────────────────────────────────────────────────────────────
    _section("Step 4 / 5  -  Training Model")
    log.info(f"  Fitting on {X_train.shape[0]:,} training samples with {X_train.shape[1]} features ...")
    log.info("  (Note: Scikit-learn MLP internally reserves 10% of training data for early stopping)")
    t0 = time.perf_counter()
    model.fit(X_train_scaled, y_train)
    elapsed = time.perf_counter() - t0
    log.info(f"  Training complete in {elapsed:.1f}s")
    if getattr(model, 'best_validation_score_', None) is not None:
        log.info(f"  Best val score achieved   : {model.best_validation_score_:.4f}")
    elif getattr(model, 'best_loss_', None) is not None:
        log.info(f"  Best loss achieved        : {model.best_loss_:.4f}")
    log.info(f"  Total epochs run          : {model.n_iter_}")

    # ── 5. Evaluate on Validation Set ────────────────────────────────────────
    _section("Step 5 / 5  -  Evaluation Results  (Explicit Validation Set)")
    
    # Output probability distributions for the ensemble
    probas = model.predict_proba(X_val_scaled)
    log.info(f"  Generated probability distribution matrix of shape: {probas.shape}")
    
    # Derive hard classifications for local evaluation metrics
    classes = model.classes_
    y_pred = classes[np.argmax(probas, axis=1)]

    classes = np.sort(np.unique(y_val))
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
    per_class_rec = recall_score(y_val, y_pred, average=None, labels=classes, zero_division=0)
    log.info("  Per-class breakdown:")
    for cls, prec, rec in zip(classes, per_class_prec, per_class_rec):
        log.info(f"    Class {cls:<2} ->  Precision: {prec:.4f}  |  Recall: {rec:.4f}")
    log.info("")

    # ── 6. Serialise Model & Scaler ───────────────────────────────────────────
    _section("Serialising Model & Scaler")
    _MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{file_prefix}mlp_model_{timestamp}.joblib"
    scaler_filename = f"{file_prefix}mlp_scaler_{timestamp}.joblib"
    
    model_path = _MODEL_OUTPUT_DIR / model_filename
    scaler_path = _MODEL_OUTPUT_DIR / scaler_filename
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    model_size_mb = model_path.stat().st_size / 1_048_576
    scaler_size_mb = scaler_path.stat().st_size / 1_048_576
    
    log.info(f"  Model saved -> {model_path} ({model_size_mb:.2f} MB)")
    log.info(f"  Scaler saved -> {scaler_path} ({scaler_size_mb:.2f} MB)")
    log.info("")
    _section("Done  [v]  mlp_trainer.py completed successfully")
    log.info("")

if __name__ == "__main__":
    main()
