"""
logreg_trainer.py
=================
Standalone Logistic Regression Classifier -- Train & Validate

Pipeline Stage : Model Training (Phase 3 -- Ensemble Member #3)
Inputs         : Any .parquet file containing 'train' / 'val' in its name
                 Auto-discovered in ../Feature Engineering/Outputs/
Outputs        : Outputs/logreg_pipeline_YYYYMMDD_HHMMSS.joblib
                 (sklearn Pipeline bundling RobustScaler + LogisticRegression)

Usage
-----
    # Auto-discover files by substring match:
    python logreg_trainer.py

    # Override one or both paths explicitly:
    python logreg_trainer.py --train_path /path/to/train.parquet
    python logreg_trainer.py --train_path /path/to/train.parquet --val_path /path/to/val.parquet

Notes
-----
  - RobustScaler is used over StandardScaler: financial data contains
    outlier spikes (vol events, gaps) that IQR-based scaling handles
    more gracefully than mean/std.
  - solver='saga' is required: the only sklearn solver that supports
    multinomial + class_weight='balanced' + n_jobs=-1 simultaneously.
    lbfgs silently ignores n_jobs for multinomial problems.
  - The scaler and model are bundled into a single sklearn Pipeline and
    saved as one .joblib artifact so the orchestrator always loads a
    consistent, matched pair.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

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

LOGREG_PARAMS: dict = {
    "solver": "saga",           # saga: best for large datasets; multinomial by default for 3+ classes
    "class_weight": "balanced",
    "max_iter": 1000,
    "random_state": 42,
    "tol": 1e-4,
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
    """Extract target vector, drop target + non-stationary features."""
    if target not in df.columns:
        log.error(f"[TARGET NOT FOUND] Column '{target}' missing in {dataset_label} dataset.")
        sys.exit(1)

    y = df[target]
    X = df.drop(columns=[target])

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
        description="Train and validate a Logistic Regression classifier for the Quant Lab ensemble.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python logreg_trainer.py\n"
            "  python logreg_trainer.py --train_path /path/to/train.parquet\n"
            "  python logreg_trainer.py --train_path /path/to/train.parquet --val_path /path/to/val.parquet"
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
    _section("LogReg Trainer  -  Quantitative Trading Pipeline  |  Phase 3")

    # ── 1. Load Data ──────────────────────────────────────────────────────────
    _section("Step 1 / 4  -  Loading Data")

    if args.train_path:
        train_file = args.train_path
        log.info(f"  Train file  : [CLI override] {train_file}")
    else:
        log.info(f"  Scanning for 'train' parquet in: {_FEATURE_OUTPUT_DIR}")
        train_file = _discover_file(_FEATURE_OUTPUT_DIR, "train")
        log.info(f"  Train file  : [auto-discovered] {train_file.name}")

    if args.val_path:
        val_file = args.val_path
        log.info(f"  Val file    : [CLI override] {val_file}")
    else:
        log.info(f"  Scanning for 'val' parquet in: {_FEATURE_OUTPUT_DIR}")
        val_file = _discover_file(_FEATURE_OUTPUT_DIR, "val")
        log.info(f"  Val file    : [auto-discovered] {val_file.name}")

    df_train = _load_parquet(train_file, "Train")
    df_val   = _load_parquet(val_file,   "Validation")

    X_train, y_train = _split_xy(df_train, TARGET_COL, "Train")
    X_val,   y_val   = _split_xy(df_val,   TARGET_COL, "Validation")

    # Align columns in case MI purge created any schema difference
    missing_in_val = set(X_train.columns) - set(X_val.columns)
    if missing_in_val:
        log.warning(f"  [COLUMN MISMATCH] Features in train but not val: {missing_in_val}")
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)

    log.info(f"  Class distribution (train): {dict(y_train.value_counts().sort_index())}")
    log.info(f"  Class distribution (val)  : {dict(y_val.value_counts().sort_index())}")

    # ── 2. Build Pipeline (Scaler + Model) ───────────────────────────────────
    _section("Step 2 / 4  -  Building Pipeline  [RobustScaler -> LogisticRegression]")

    log.info("  Scaler  : RobustScaler")
    log.info("            Uses median/IQR -- robust to financial outlier spikes")
    log.info("")
    log.info("  Model   : LogisticRegression")
    for k, v in LOGREG_PARAMS.items():
        log.info(f"    {k:<18}: {v}")
    log.info("")

    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("model",  LogisticRegression(**LOGREG_PARAMS)),
    ])

    # ── 3. Train ──────────────────────────────────────────────────────────────
    _section("Step 3 / 4  -  Training  (Scaler fit on train only)")
    log.info(
        f"  Fitting on {X_train.shape[0]:,} training samples "
        f"with {X_train.shape[1]} features ..."
    )
    log.info("  [Scaler will be fit on X_train; X_val receives transform only]")
    log.info("")

    t0 = time.perf_counter()
    pipe.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    lr_model = pipe.named_steps["model"]
    converged = lr_model.n_iter_[0] < LOGREG_PARAMS["max_iter"]
    log.info(f"  Training complete in {elapsed:.1f}s")
    log.info(f"  Iterations used    : {lr_model.n_iter_[0]} / {LOGREG_PARAMS['max_iter']}")
    log.info(
        f"  Convergence        : {'YES' if converged else 'NO -- consider increasing max_iter'}"
    )

    # ── 4. Evaluate on Validation Set ────────────────────────────────────────
    _section("Step 4 / 4  -  Evaluation Results  (Validation Set)")
    y_pred = pipe.predict(X_val)

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

    # Absolute coefficient analysis -- averaged across all class OvR vectors
    # coef_ shape: (n_classes, n_features)
    # Mean absolute coef across classes shows overall linear importance.
    mean_abs_coef = pd.Series(
        np.abs(lr_model.coef_).mean(axis=0),
        index=X_train.columns,
    ).sort_values(ascending=False)

    log.info("  Feature Coefficients  (mean |coef| across all classes):")
    max_coef = mean_abs_coef.iloc[0] if mean_abs_coef.iloc[0] > 0 else 1.0
    for feat, coef in mean_abs_coef.items():
        bar = "=" * int((coef / max_coef) * 40)
        log.info(f"    {feat:<25} {coef:>8.4f}  {bar}")
    log.info("")

    # Per-class coefficient breakdown for transparency
    log.info("  Per-class top coefficients  (signed, post-scaling):")
    for i, cls in enumerate(lr_model.classes_):
        coefs = pd.Series(lr_model.coef_[i], index=X_train.columns)
        top3_pos = coefs.nlargest(3)
        top3_neg = coefs.nsmallest(3)
        log.info(f"    Class {cls}  ->  "
                 f"Most positive: {dict(top3_pos.round(4))}  |  "
                 f"Most negative: {dict(top3_neg.round(4))}")
    log.info("")

    # ── 5. Serialise Pipeline ─────────────────────────────────────────────────
    _section("Serialising Pipeline  [RobustScaler + LogisticRegression]")
    _MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
    pipeline_filename = f"logreg_pipeline_{timestamp}.joblib"
    pipeline_path   = _MODEL_OUTPUT_DIR / pipeline_filename

    joblib.dump(pipe, pipeline_path)
    size_mb = pipeline_path.stat().st_size / 1_048_576
    log.info(f"  Pipeline saved -> {pipeline_path}")
    log.info(f"  File size      : {size_mb:.2f} MB")
    log.info(f"  Contents       : RobustScaler + LogisticRegression (bundled)")
    log.info(f"  Usage          : pipe = joblib.load(path); pipe.predict(X_raw)")
    log.info("")
    _section("Done  [v]  logreg_trainer.py completed successfully")
    log.info("")


if __name__ == "__main__":
    main()
