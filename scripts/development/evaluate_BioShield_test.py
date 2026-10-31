import numpy as np
import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    brier_score_loss,
    confusion_matrix,
)

RANDOM_STATE = 42
N_OOF_FOLDS = 5
THRESHOLD = 0.4950

SPLIT_FILE = "bioshield_v2_repaired_split_assignments_BioShield.csv"
INDEX_FILE = "bioshield_feature_index.csv"

PHYS_FILE = "BioShield_features/phys_BioShield.npy"
ESM35_FILE = "BioShield_features/esm35_BioShield.npy"


# ============================================================
# 1. LOAD AUTHORITATIVE BioShield DATA
# ============================================================

split = pd.read_csv(SPLIT_FILE)
idx = pd.read_csv(INDEX_FILE)

phys = np.load(PHYS_FILE)
esm35 = np.load(ESM35_FILE)

assert len(idx) == 9512
assert phys.shape == (9512, 28)
assert esm35.shape == (9512, 480)

assert idx["feature_row"].tolist() == list(range(9512))

# The BioShield split file contains the authoritative family assignment.
# Merge family information onto the feature index.
family_map = split[
    ["v2_id", "component"]
].drop_duplicates("v2_id")

assert len(family_map) == 9512

data = idx.merge(
    family_map,
    on="v2_id",
    how="left",
    validate="one_to_one",
)

assert data["component"].notna().all()

X_all = np.concatenate(
    [
        np.asarray(phys, dtype=np.float32),
        np.asarray(esm35, dtype=np.float32),
    ],
    axis=1,
)

assert X_all.shape == (9512, 508)


# ============================================================
# 2. TRAIN / TEST ROWS
# ============================================================

train_mask = data["split_final_BioShield"].eq("train")
test_mask = data["split_final_BioShield"].eq("test")

train_rows = data.loc[
    train_mask, "feature_row"
].to_numpy()

test_rows = data.loc[
    test_mask, "feature_row"
].to_numpy()

y_train = data.loc[
    train_mask, "label"
].astype(int).to_numpy()

y_test = data.loc[
    test_mask, "label"
].astype(int).to_numpy()

groups_train = data.loc[
    train_mask, "component"
].astype(str).to_numpy()

X_train = X_all[train_rows]
X_test = X_all[test_rows]

assert X_train.shape == (6658, 508)
assert X_test.shape == (1427, 508)
assert len(y_train) == 6658
assert len(y_test) == 1427

assert len(np.unique(groups_train)) == 6119


print("=" * 90)
print("BioShield FINAL BLIND TEST — FUSION ESM35")
print("=" * 90)

print("\nTRAIN:", X_train.shape)
print("TEST :", X_test.shape)

print("\nTRAIN labels:")
print(pd.Series(y_train).value_counts().sort_index())

print("\nTEST labels:")
print(pd.Series(y_test).value_counts().sort_index())

print("\nTraining families:", len(np.unique(groups_train)))


# ============================================================
# 3. MODEL DEFINITION
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
# 4. GROUP-AWARE TRAIN OOF
#
# This reproduces the TRAIN-derived calibration procedure.
# TEST is not involved.
# ============================================================

cv = StratifiedGroupKFold(
    n_splits=N_OOF_FOLDS,
    shuffle=True,
    random_state=RANDOM_STATE,
)

oof_prob = np.zeros(
    len(y_train),
    dtype=np.float64,
)

for fold, (tr, va) in enumerate(
    cv.split(X_train, y_train, groups_train),
    start=1,
):

    print(
        f"\nCalibration OOF fold {fold}/{N_OOF_FOLDS}: "
        f"train={len(tr)}, val={len(va)}"
    )

    train_groups = set(groups_train[tr])
    valid_groups = set(groups_train[va])

    overlap = train_groups.intersection(valid_groups)

    if overlap:
        raise RuntimeError(
            f"Family leakage in OOF fold {fold}: "
            f"{len(overlap)} overlapping families"
        )

    model = make_model()

    model.fit(
        X_train[tr],
        y_train[tr],
    )

    oof_prob[va] = model.predict_proba(
        X_train[va]
    )[:, 1]


assert np.isfinite(oof_prob).all()


# ============================================================
# 5. PLATT CALIBRATION
#
# Calibration is learned ONLY from TRAIN OOF predictions.
# ============================================================

eps = 1e-6

p = np.clip(
    oof_prob,
    eps,
    1 - eps,
)

logit_p = np.log(
    p / (1 - p)
).reshape(-1, 1)

calibrator = LogisticRegression(
    solver="lbfgs",
    max_iter=1000,
    random_state=RANDOM_STATE,
)

calibrator.fit(
    logit_p,
    y_train,
)

oof_cal_prob = calibrator.predict_proba(
    logit_p
)[:, 1]

assert np.isfinite(oof_cal_prob).all()


# ============================================================
# 6. VERIFY LOCKED THRESHOLD
# ============================================================

print("\nLocked threshold:", THRESHOLD)

oof_pred = (
    oof_cal_prob >= THRESHOLD
).astype(int)

oof_mcc = matthews_corrcoef(
    y_train,
    oof_pred,
)

print(
    "TRAIN OOF MCC at locked threshold:",
    f"{oof_mcc:.6f}"
)


# ============================================================
# 7. FINAL MODEL — TRAIN ONLY
# ============================================================

print("\nFitting final XGBoost model on TRAIN only...")

final_model = make_model()

final_model.fit(
    X_train,
    y_train,
)


# ============================================================
# 8. SINGLE BLIND TEST PREDICTION
# ============================================================

print("\nGenerating TEST predictions...")

raw_test_prob = final_model.predict_proba(
    X_test
)[:, 1]

p_test = np.clip(
    raw_test_prob,
    eps,
    1 - eps,
)

test_logit = np.log(
    p_test / (1 - p_test)
).reshape(-1, 1)

cal_test_prob = calibrator.predict_proba(
    test_logit
)[:, 1]

test_pred = (
    cal_test_prob >= THRESHOLD
).astype(int)


# ============================================================
# 9. TEST METRICS
# ============================================================

roc_auc = roc_auc_score(
    y_test,
    cal_test_prob,
)

pr_auc = average_precision_score(
    y_test,
    cal_test_prob,
)

accuracy = accuracy_score(
    y_test,
    test_pred,
)

precision = precision_score(
    y_test,
    test_pred,
    zero_division=0,
)

sensitivity = recall_score(
    y_test,
    test_pred,
    zero_division=0,
)

f1 = f1_score(
    y_test,
    test_pred,
    zero_division=0,
)

mcc = matthews_corrcoef(
    y_test,
    test_pred,
)

brier = brier_score_loss(
    y_test,
    cal_test_prob,
)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    test_pred,
).ravel()

specificity = tn / (tn + fp)


# ============================================================
# 10. PRINT FINAL RESULTS
# ============================================================

print("\n" + "=" * 90)
print("FINAL BioShield TEST RESULTS")
print("=" * 90)

print(f"ROC-AUC       : {roc_auc:.6f}")
print(f"PR-AUC        : {pr_auc:.6f}")
print(f"Accuracy      : {accuracy:.6f}")
print(f"Precision     : {precision:.6f}")
print(f"Sensitivity   : {sensitivity:.6f}")
print(f"Specificity   : {specificity:.6f}")
print(f"F1            : {f1:.6f}")
print(f"MCC           : {mcc:.6f}")
print(f"Brier         : {brier:.6f}")

print(f"\nThreshold     : {THRESHOLD:.6f}")

print("\nConfusion matrix:")
print("[[TN, FP],")
print(" [FN, TP]]")
print(
    np.array([
        [tn, fp],
        [fn, tp],
    ])
)


# ============================================================
# 11. SAVE TEST RESULTS
# ============================================================

test_out = data.loc[
    test_mask,
    [
        "v2_id",
        "sequence",
        "label",
        "split_final_BioShield",
    ],
].copy()

test_out["raw_probability"] = raw_test_prob
test_out["calibrated_probability"] = cal_test_prob
test_out["prediction"] = test_pred

test_out.to_csv(
    "BioShield_final_test_predictions.csv",
    index=False,
)

metrics_out = pd.DataFrame([{
    "feature_set": "fusion_esm35",
    "probability_type": "platt",
    "threshold": THRESHOLD,
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "accuracy": accuracy,
    "precision": precision,
    "sensitivity": sensitivity,
    "specificity": specificity,
    "f1": f1,
    "mcc": mcc,
    "brier": brier,
    "tn": tn,
    "fp": fp,
    "fn": fn,
    "tp": tp,
}])

metrics_out.to_csv(
    "BioShield_final_test_metrics.csv",
    index=False,
)

print("\nSaved:")
print("  BioShield_final_test_predictions.csv")
print("  BioShield_final_test_metrics.csv")

print("\n" + "=" * 90)
print("TEST EVALUATION COMPLETE")
print("Configuration was locked before TEST evaluation.")
print("=" * 90)
