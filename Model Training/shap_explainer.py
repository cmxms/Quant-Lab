"""
shap_explainer.py
=================
SHAP "Lie Detector" -- Audit Trained Tree Models

Pipeline Stage : Model Audit (Phase 3 -- Post-Training)
Inputs         : Most recent lgbm_model_*.joblib and rf_model_*.joblib
                 from Outputs/ directory.
                 long_clean_val.parquet from ../Feature Engineering/Outputs/
Outputs        : Outputs/shap_summary_bar_lgbm.png
                 Outputs/shap_summary_dot_lgbm.png
                 Outputs/shap_dependence_lgbm.png
                 Outputs/shap_summary_bar_rf.png
                 Outputs/shap_summary_dot_rf.png
                 Outputs/shap_dependence_rf.png

Purpose
-------
SHAP (SHapley Additive exPlanations) audits whether each model is using
features for the *reasons we expect*. Mismatches between SHAP rankings
and domain intuition expose potential leakage, spurious correlations, or
features the model has latched onto for wrong reasons ("the lie").

Usage
-----
    python shap_explainer.py
    python shap_explainer.py --n_samples 2000
    python shap_explainer.py --val_path /path/to/val.parquet
"""

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend -- must be set before pyplot import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

# ---------------------------------------------------------------------------
# ── Configuration ───────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent.resolve()
_FEATURE_OUTPUT_DIR = _HERE.parent / "Feature Engineering" / "Outputs"
_MODEL_OUTPUT_DIR   = _HERE / "Outputs"

TARGET_COL: str = "Target"
DROP_COLS: list = [
    "Open", "High", "Low", "Close", "Volume",
    "Minutes_From_Open", "Day_Of_Week",
]

DEPENDENCE_FEATURE: str = "RVOL_60m"   # Feature to spotlight in dependence plots
N_SAMPLES_DEFAULT:  int = 1000         # Background sample size for SHAP
SHAP_CLASS_IDX:     int = 2            # Class index for dependence plot (0=-1, 1=0, 2=1)
                                       # Index 2 = Class 1 (long signal) -- most actionable

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
# ── Helper: Find Most Recent Model File ─────────────────────────────────────
# ---------------------------------------------------------------------------

def _find_latest_model(directory: Path, pattern: str) -> Path:
    """Return the most recently modified .joblib matching pattern."""
    matches = sorted(directory.glob(pattern), key=lambda p: p.stat().st_mtime)
    if not matches:
        log.error(f"[NO MODEL] No file matching '{pattern}' found in: {directory}")
        sys.exit(1)
    latest = matches[-1]
    log.info(f"  Found {len(matches)} match(es). Using most recent: {latest.name}")
    return latest


# ---------------------------------------------------------------------------
# ── Helper: Prepare Feature Matrix ──────────────────────────────────────────
# ---------------------------------------------------------------------------

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop Target and non-stationary columns, return clean feature DataFrame."""
    cols_to_drop = [c for c in [TARGET_COL] + DROP_COLS if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    log.info(f"  Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")
    log.info(f"  Columns: {list(X.columns)}")
    return X


# ---------------------------------------------------------------------------
# ── Helper: Normalise SHAP Values to 3D Array ───────────────────────────────
# ---------------------------------------------------------------------------

def _to_3d_shap(shap_values, n_classes: int, n_samples: int, n_features: int) -> np.ndarray:
    """
    Normalise SHAP output to a consistent (n_samples, n_features, n_classes) array.

    TreeExplainer returns different shapes depending on the model and shap version:
      - LGBM multiclass : (n_samples, n_features, n_classes)  [3D already]
      - RF multiclass   : list of n_classes arrays, each (n_samples, n_features)
      - Binary          : (n_samples, n_features)              [2D]
    """
    if isinstance(shap_values, list):
        # List of 2D arrays -- stack along new axis 2
        return np.stack(shap_values, axis=2)
    arr = np.array(shap_values)
    if arr.ndim == 3:
        return arr          # Already (n_samples, n_features, n_classes)
    if arr.ndim == 2:
        return arr[:, :, np.newaxis]  # Binary edge case
    raise ValueError(f"Unexpected SHAP values shape: {arr.shape}")


# ---------------------------------------------------------------------------
# ── Core: Compute & Plot SHAP for One Model ─────────────────────────────────
# ---------------------------------------------------------------------------

def _run_shap(
    model_path: Path,
    model_label: str,      # e.g. "lgbm" or "rf"
    X_sample: pd.DataFrame,
    class_names: list,
) -> None:
    """Load model, compute SHAP values, save bar/dot summary and dependence plots."""

    _section(f"SHAP Audit  --  {model_label.upper()}")

    # ── Load model ────────────────────────────────────────────────────────────
    log.info(f"  Loading model: {model_path.name}")
    model = joblib.load(model_path)

    # Unwrap Pipeline if necessary (logreg uses Pipeline; tree models don't)
    if hasattr(model, "named_steps"):
        model = model.named_steps.get("model", model)

    # ── Build TreeExplainer ───────────────────────────────────────────────────
    log.info(f"  Building shap.TreeExplainer on {X_sample.shape[0]:,} sample rows ...")
    explainer = shap.TreeExplainer(model)

    log.info("  Computing SHAP values (this may take 30-120s) ...")
    raw_shap = explainer.shap_values(X_sample)

    n_classes  = len(class_names)
    n_samples  = X_sample.shape[0]
    n_features = X_sample.shape[1]

    shap_3d = _to_3d_shap(raw_shap, n_classes, n_samples, n_features)
    # shap_3d shape: (n_samples, n_features, n_classes)

    log.info(f"  SHAP values shape: {shap_3d.shape}  "
             f"(samples x features x classes)")

    # Mean absolute SHAP across all classes -- global feature impact
    mean_abs_shap = np.abs(shap_3d).mean(axis=(0, 2))   # (n_features,)
    importance_series = pd.Series(mean_abs_shap, index=X_sample.columns)\
                          .sort_values(ascending=False)

    log.info("  Global mean |SHAP| per feature:")
    for feat, val in importance_series.items():
        bar = "=" * int((val / importance_series.iloc[0]) * 40)
        log.info(f"    {feat:<25} {val:.5f}  {bar}")
    log.info("")

    # ── Plot 1: Bar Summary (mean |SHAP| across all classes) ─────────────────
    bar_path = _MODEL_OUTPUT_DIR / f"shap_summary_bar_{model_label}.png"
    log.info(f"  Saving bar summary -> {bar_path.name}")

    fig, ax = plt.subplots(figsize=(10, 7))
    importance_series.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_xlabel("Mean |SHAP Value|  (avg across all classes)", fontsize=11)
    ax.set_title(
        f"SHAP Global Feature Importance  [{model_label.upper()}]\n"
        f"(1,000-row val sample, mean |SHAP| averaged across {n_classes} classes)",
        fontsize=12,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(bar_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {bar_path}")

    # ── Plot 2: Beeswarm/Dot Summary (Class 1 = long signal) ─────────────────
    dot_path = _MODEL_OUTPUT_DIR / f"shap_summary_dot_{model_label}.png"
    log.info(f"  Saving dot summary (Class 1 = long signal) -> {dot_path.name}")

    shap_class1 = shap_3d[:, :, SHAP_CLASS_IDX]   # (n_samples, n_features)

    fig, ax = plt.subplots(figsize=(10, 8))
    shap.summary_plot(
        shap_class1,
        X_sample,
        plot_type="dot",
        show=False,
        title=f"SHAP Beeswarm  [{model_label.upper()}]  --  Class 1 (Long Signal)",
        max_display=14,
    )
    plt.title(
        f"SHAP Beeswarm  [{model_label.upper()}]  --  Class 1 (Long Signal)\n"
        f"Red = high feature value pushes toward long prediction",
        fontsize=11,
    )
    plt.tight_layout()
    fig.savefig(dot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved: {dot_path}")

    # ── Plot 3: Dependence Plot (RVOL_60m, Class 1) ───────────────────────────
    dep_path = _MODEL_OUTPUT_DIR / f"shap_dependence_{model_label}.png"
    log.info(f"  Saving dependence plot ({DEPENDENCE_FEATURE}) -> {dep_path.name}")

    if DEPENDENCE_FEATURE not in X_sample.columns:
        log.warning(
            f"  [SKIP] '{DEPENDENCE_FEATURE}' not found in feature set. "
            f"Available: {list(X_sample.columns)}"
        )
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.dependence_plot(
            DEPENDENCE_FEATURE,
            shap_class1,
            X_sample,
            interaction_index="auto",   # SHAP auto-selects best interacting feature
            ax=ax,
            show=False,
            alpha=0.5,
        )
        ax.set_title(
            f"SHAP Dependence: {DEPENDENCE_FEATURE}  [{model_label.upper()}]\n"
            f"Class 1 (Long Signal)  --  color = auto-selected interaction feature",
            fontsize=11,
        )
        plt.tight_layout()
        fig.savefig(dep_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        log.info(f"  Saved: {dep_path}")

    log.info("")


# ---------------------------------------------------------------------------
# ── Main ────────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="SHAP Lie Detector -- Audit trained tree models with SHAP values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python shap_explainer.py\n"
            "  python shap_explainer.py --n_samples 2000\n"
            "  python shap_explainer.py --val_path /path/to/val.parquet"
        ),
    )
    parser.add_argument(
        "--val_path",
        type=Path,
        default=None,
        help="Override: explicit path to validation .parquet file.",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=N_SAMPLES_DEFAULT,
        help=f"Number of rows to sample from val set for SHAP (default: {N_SAMPLES_DEFAULT}).",
    )
    args = parser.parse_args()

    log.info("")
    _section("SHAP Explainer  -  Quantitative Trading Pipeline  |  Model Audit")

    # ── 1. Locate Models ──────────────────────────────────────────────────────
    _section("Step 1 / 3  -  Locating Most Recent Models")
    lgbm_path = _find_latest_model(_MODEL_OUTPUT_DIR, "lgbm_model_*.joblib")
    rf_path   = _find_latest_model(_MODEL_OUTPUT_DIR, "rf_model_*.joblib")

    # ── 2. Load & Sample Validation Data ─────────────────────────────────────
    _section("Step 2 / 3  -  Loading & Sampling Validation Data")

    val_file = args.val_path or (_FEATURE_OUTPUT_DIR / "long_clean_val.parquet")
    if not val_file.exists():
        log.error(f"[MISSING FILE] Validation file not found: {val_file}")
        sys.exit(1)

    log.info(f"  Reading: {val_file.name}")
    df_val = pd.read_parquet(val_file)
    log.info(f"  Full val shape: {df_val.shape[0]:,} rows x {df_val.shape[1]} columns")

    # Extract class names before dropping Target
    classes = sorted(df_val[TARGET_COL].unique().tolist()) if TARGET_COL in df_val.columns else [-1, 0, 1]
    class_names = [str(c) for c in classes]

    X_full = _prepare_features(df_val)

    n_samples = min(args.n_samples, len(X_full))
    log.info(f"  Sampling {n_samples:,} rows (random_state=42) ...")
    X_sample = X_full.sample(n=n_samples, random_state=42).reset_index(drop=True)
    log.info(f"  Sample shape: {X_sample.shape}")
    log.info(f"  Classes: {class_names}")

    # ── 3. SHAP Audit -- Both Models ─────────────────────────────────────────
    _section("Step 3 / 3  -  Computing SHAP Values  (LGBM + RF)")
    _MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    _run_shap(lgbm_path, "lgbm", X_sample, class_names)
    _run_shap(rf_path,   "rf",   X_sample, class_names)

    # ── Summary ───────────────────────────────────────────────────────────────
    _section("All SHAP Plots Generated  [v]")
    output_files = [
        "shap_summary_bar_lgbm.png",
        "shap_summary_dot_lgbm.png",
        "shap_dependence_lgbm.png",
        "shap_summary_bar_rf.png",
        "shap_summary_dot_rf.png",
        "shap_dependence_rf.png",
    ]
    for fname in output_files:
        fpath = _MODEL_OUTPUT_DIR / fname
        if fpath.exists():
            size_kb = fpath.stat().st_size / 1024
            log.info(f"  {fname:<40} {size_kb:>7.1f} KB")
        else:
            log.warning(f"  {fname:<40} [NOT FOUND]")
    log.info("")
    _section("Done  [v]  shap_explainer.py completed successfully")
    log.info("")


if __name__ == "__main__":
    main()
