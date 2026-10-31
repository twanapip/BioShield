#!/usr/bin/env python

import os
import json
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

RANDOM_STATE = 42
N_OUTER = 5
N_INNER = 3

BASE = "."

INDEX_FILE = "bioshield_feature_index.csv"
FAMILY_FILE = "BioShield_similarity_families.csv"

FEATURE_DIR = "BioShield_features"
OUT_DIR = "BioShield_nested_cv"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD AUTHORITATIVE INDEX + FAMILY ASSIGNMENTS
# ============================================================

idx = pd.read_csv(INDEX_FILE)
fam = pd.read_csv(FAMILY_FILE)

required_idx = [
    "v2_id",
    "sequence",
    "label",
    "split_final_BioShield",
]

required_fam = [
    "v2_id",
    "family",
]

for c in required_idx:
    if c not in idx.columns:
        raise ValueError(f"Missing index column: {c}")

for c in required_fam:
    if c not in fam.columns:
        raise ValueError(f"Missing family column: {c}")

d = idx.merge(
    fam[["v2_id", "family"]],
    on="v2_id",
    how="left",
    validate="one_to_one",
)

if d["family"].isna().any():
    raise ValueError("Some sequences have no family assignment.")

# ============================================================
# BASIC FROZEN-SPLIT CHECKS
# ============================================================

assert d["v2_id"].nunique() == len(d)
assert d["sequence"].nunique() == len(d)

assert set(d["split_final_BioShield"]) == {"train", "val", "test"}

train = d[d["split_final_BioShield"] == "train"].copy()
val = d[d["split_final_BioShield"] == "val"].copy()
test = d[d["split_final_BioShield"] == "test"].copy()

print("=" * 80)
print("BioShield NESTED GROUP-AWARE CV")
print("=" * 80)

print("\nFrozen split:")
print(d["split_final_BioShield"].value_counts())

print("\nTrain labels:")
print(train["label"].value_counts().sort_index())

print("\nTrain families:", train["family"].nunique())
print("Val families:", val["family"].nunique())
print("Test families:", test["family"].nunique())

# Verify no family crosses frozen splits.
split_per_family = d.groupby("family")["split_final_BioShield"].nunique()

if (split_per_family > 1).any():
    bad = split_per_family[split_per_family > 1]
    raise ValueError(
        f"Family crosses frozen split boundaries: {len(bad)} families"
    )

print("\nFrozen family boundary check: PASS")

# ============================================================
# LOAD FEATURES
# ============================================================

feature_files = {
    "phys": "phys_BioShield.npy",
    "esm8": "esm8_BioShield.npy",
    "esm35": "esm35_BioShield.npy",
    "esm150": "esm150_BioShield.npy",
}

features = {}

for name, fn in feature_files.items():

    path = os.path.join(FEATURE_DIR, fn)

    print(f"\nLoading {name}: {path}")

    X = np.load(path, mmap_mode="r")

    if len(X) != len(d):
        raise ValueError(
            f"{name}: feature rows {len(X)} != index rows {len(d)}"
        )

    if not np.isfinite(X).all():
        raise ValueError(f"{name}: NaN/Inf detected")

    features[name] = X

    print("shape:", X.shape)
    print("dtype:", X.dtype)

# ============================================================
# FEATURE SETS
# ============================================================

feature_sets = {
    "phys": ["phys"],
    "esm8": ["esm8"],
    "esm35": ["esm35"],
    "esm150": ["esm150"],
    "fusion_esm8": ["phys", "esm8"],
    "fusion_esm35": ["phys", "esm35"],
    "fusion_esm150": ["phys", "esm150"],
}

def make_X(names, rows):

    matrices = [
        np.asarray(features[n][rows], dtype=np.float32)
        for n in names
    ]

    return np.concatenate(matrices, axis=1)


# ============================================================
# MODEL
# ============================================================

def make_model():

    return XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.80,
        colsample_bytree=0.80,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y, prob, threshold=0.5):

    pred = (prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        pred,
        labels=[0, 1],
    ).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "roc_auc": roc_auc_score(y, prob),
        "pr_auc": average_precision_score(y, prob),
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(
            y, pred, zero_division=0
        ),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1_score(
            y, pred, zero_division=0
        ),
        "mcc": matthews_corrcoef(y, pred),
        "brier": brier_score_loss(y, prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# PLATT CALIBRATION
# ============================================================

def fit_platt(y, prob):

    eps = 1e-6

    prob = np.clip(prob, eps, 1 - eps)

    logit = np.log(prob / (1 - prob))

    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )

    calibrator.fit(
        logit.reshape(-1, 1),
        y,
    )

    return calibrator


def apply_platt(calibrator, prob):

    eps = 1e-6

    prob = np.clip(prob, eps, 1 - eps)

    logit = np.log(prob / (1 - prob))

    return calibrator.predict_proba(
        logit.reshape(-1, 1)
    )[:, 1]


# ============================================================
# THRESHOLD OPTIMIZATION
# ============================================================

def choose_threshold(y, prob):

    thresholds = np.linspace(
        0.05,
        0.95,
        181,
    )

    best_threshold = 0.5
    best_mcc = -np.inf

    for t in thresholds:

        pred = (prob >= t).astype(int)

        mcc = matthews_corrcoef(y, pred)

        if mcc > best_mcc:

            best_mcc = mcc
            best_threshold = t

    return best_threshold


# ============================================================
# NESTED CV
# ============================================================

all_outer_metrics = []
all_outer_predictions = []
all_inner_results = []

train_indices = np.arange(len(train))

y_train_all = train["label"].to_numpy()
groups_train_all = train["family"].to_numpy()

print("\n" + "=" * 80)
print("STARTING NESTED CV")
print("=" * 80)

for feature_name, feature_components in feature_sets.items():

    print("\n" + "#" * 80)
    print("FEATURE SET:", feature_name)
    print("#" * 80)

    X_all = make_X(
        feature_components,
        train_indices,
    )

    print("Feature matrix:", X_all.shape)

    outer_cv = StratifiedGroupKFold(
        n_splits=N_OUTER,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    for outer_fold, (outer_tr, outer_te) in enumerate(
        outer_cv.split(
            X_all,
            y_train_all,
            groups_train_all,
        ),
        start=1,
    ):

        print(
            f"\n{feature_name} | OUTER FOLD {outer_fold}/{N_OUTER}"
        )

        X_outer_train = X_all[outer_tr]
        y_outer_train = y_train_all[outer_tr]
        g_outer_train = groups_train_all[outer_tr]

        X_outer_test = X_all[outer_te]
        y_outer_test = y_train_all[outer_te]

        # ----------------------------------------------------
        # INNER CV
        # ----------------------------------------------------

        inner_cv = StratifiedGroupKFold(
            n_splits=N_INNER,
            shuffle=True,
            random_state=RANDOM_STATE + outer_fold,
        )

        inner_oof = np.zeros(
            len(outer_tr),
            dtype=np.float64,
        )

        for inner_fold, (inner_tr, inner_va) in enumerate(
            inner_cv.split(
                X_outer_train,
                y_outer_train,
                g_outer_train,
            ),
            start=1,
        ):

            print(
                f"  Inner fold {inner_fold}/{N_INNER}"
            )

            model = make_model()

            model.fit(
                X_outer_train[inner_tr],
                y_outer_train[inner_tr],
            )

            inner_oof[inner_va] = model.predict_proba(
                X_outer_train[inner_va]
            )[:, 1]

        # ----------------------------------------------------
        # INNER OOF THRESHOLD
        # ----------------------------------------------------

        threshold = choose_threshold(
            y_outer_train,
            inner_oof,
        )

        inner_metrics = calculate_metrics(
            y_outer_train,
            inner_oof,
            threshold,
        )

        inner_metrics.update({
            "feature_set": feature_name,
            "outer_fold": outer_fold,
            "threshold": threshold,
        })

        all_inner_results.append(inner_metrics)

        print(
            f"  Inner OOF ROC-AUC: {inner_metrics['roc_auc']:.4f}"
        )
        print(
            f"  Inner OOF PR-AUC:  {inner_metrics['pr_auc']:.4f}"
        )
        print(
            f"  Inner threshold:   {threshold:.3f}"
        )
        print(
            f"  Inner MCC:         {inner_metrics['mcc']:.4f}"
        )

        # ----------------------------------------------------
        # CALIBRATION
        #
        # Calibration is fitted ONLY using inner OOF
        # predictions from the outer training partition.
        # ----------------------------------------------------

        calibrator = fit_platt(
            y_outer_train,
            inner_oof,
        )

        # ----------------------------------------------------
        # FIT OUTER MODEL
        # ----------------------------------------------------

        outer_model = make_model()

        outer_model.fit(
            X_outer_train,
            y_outer_train,
        )

        raw_outer_prob = outer_model.predict_proba(
            X_outer_test
        )[:, 1]

        calibrated_outer_prob = apply_platt(
            calibrator,
            raw_outer_prob,
        )

        # ----------------------------------------------------
        # OUTER TEST METRICS
        # ----------------------------------------------------

        raw_metrics = calculate_metrics(
            y_outer_test,
            raw_outer_prob,
            threshold,
        )

        cal_metrics = calculate_metrics(
            y_outer_test,
            calibrated_outer_prob,
            threshold,
        )

        raw_metrics.update({
            "feature_set": feature_name,
            "outer_fold": outer_fold,
            "probability_type": "raw",
            "threshold": threshold,
        })

        cal_metrics.update({
            "feature_set": feature_name,
            "outer_fold": outer_fold,
            "probability_type": "platt",
            "threshold": threshold,
        })

        all_outer_metrics.extend([
            raw_metrics,
            cal_metrics,
        ])

        # ----------------------------------------------------
        # SAVE OUTER PREDICTIONS
        # ----------------------------------------------------

        outer_ids = train.iloc[
            train_indices[outer_te]
        ]["v2_id"].to_numpy()

        outer_pred = pd.DataFrame({
            "v2_id": outer_ids,
            "label": y_outer_test,
            "raw_probability": raw_outer_prob,
            "calibrated_probability": calibrated_outer_prob,
            "threshold": threshold,
            "feature_set": feature_name,
            "outer_fold": outer_fold,
        })

        all_outer_predictions.append(
            outer_pred
        )

        print(
            f"  Outer raw ROC-AUC: "
            f"{raw_metrics['roc_auc']:.4f}"
        )

        print(
            f"  Outer raw PR-AUC:  "
            f"{raw_metrics['pr_auc']:.4f}"
        )

        print(
            f"  Outer raw MCC:     "
            f"{raw_metrics['mcc']:.4f}"
        )

        print(
            f"  Outer calibrated Brier: "
            f"{cal_metrics['brier']:.4f}"
        )


# ============================================================
# SAVE RESULTS
# ============================================================

metrics_df = pd.DataFrame(
    all_outer_metrics
)

pred_df = pd.concat(
    all_outer_predictions,
    ignore_index=True,
)

inner_df = pd.DataFrame(
    all_inner_results
)

metrics_df.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_nested_outer_metrics.csv",
    ),
    index=False,
)

pred_df.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_nested_outer_predictions.csv",
    ),
    index=False,
)

inner_df.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_nested_inner_results.csv",
    ),
    index=False,
)

# ============================================================
# SUMMARY
# ============================================================

summary_rows = []

for feature_name in feature_sets:

    for prob_type in ["raw", "platt"]:

        x = metrics_df[
            (metrics_df["feature_set"] == feature_name)
            &
            (metrics_df["probability_type"] == prob_type)
        ]

        if len(x) == 0:
            continue

        row = {
            "feature_set": feature_name,
            "probability_type": prob_type,
        }

        for metric in [
            "roc_auc",
            "pr_auc",
            "accuracy",
            "precision",
            "sensitivity",
            "specificity",
            "f1",
            "mcc",
            "brier",
        ]:

            row[f"{metric}_mean"] = x[metric].mean()
            row[f"{metric}_std"] = x[metric].std(ddof=1)

        summary_rows.append(row)

summary = pd.DataFrame(
    summary_rows
)

summary.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_nested_cv_summary.csv",
    ),
    index=False,
)

# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("BioShield NESTED CV FINAL SUMMARY")
print("=" * 100)

display_cols = [
    "feature_set",
    "probability_type",
    "roc_auc_mean",
    "roc_auc_std",
    "pr_auc_mean",
    "pr_auc_std",
    "mcc_mean",
    "mcc_std",
    "f1_mean",
    "f1_std",
    "brier_mean",
    "brier_std",
]

print(
    summary[
        display_cols
    ].sort_values(
        ["probability_type", "roc_auc_mean"],
        ascending=[True, False],
    ).to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}",
    )
)

print("\nResults saved to:")
print(OUT_DIR)

print("\nFiles:")
print("  BioShield_nested_outer_metrics.csv")
print("  BioShield_nested_outer_predictions.csv")
print("  BioShield_nested_inner_results.csv")
print("  BioShield_nested_cv_summary.csv")

print("\n" + "=" * 100)
print("NESTED CV COMPLETE")
print("=" * 100)
