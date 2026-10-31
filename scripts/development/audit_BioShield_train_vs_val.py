#!/usr/bin/env python

import os
import warnings
import numpy as np
import pandas as pd

from scipy.stats import (
    ks_2samp,
    wasserstein_distance,
)

warnings.filterwarnings("ignore")

# ============================================================
# CONFIG
# ============================================================

INDEX_FILE = "bioshield_feature_index.csv"
FAMILY_FILE = "BioShield_similarity_families.csv"
GRAPH_FILE = "bioshield_v2_complete_graph_audit.csv"

FEATURE_DIR = "BioShield_features"
OUT_DIR = "BioShield_distribution_audit"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# LOAD AUTHORITATIVE DATA
# ============================================================

idx = pd.read_csv(INDEX_FILE)
fam = pd.read_csv(FAMILY_FILE)

d = idx.merge(
    fam[["v2_id", "family"]],
    on="v2_id",
    how="left",
    validate="one_to_one",
)

assert len(d) == 9512
assert d["v2_id"].nunique() == 9512

train = d[d["split_final_BioShield"] == "train"].copy()
val = d[d["split_final_BioShield"] == "val"].copy()

# TEST IS NEVER LOADED.

print("=" * 100)
print("BioShield TRAIN vs VALIDATION DISTRIBUTION AUDIT")
print("=" * 100)

print("\nTRAIN")
print("Sequences:", len(train))
print("Families:", train["family"].nunique())
print("Label 0:", (train["label"] == 0).sum())
print("Label 1:", (train["label"] == 1).sum())

print("\nVAL")
print("Sequences:", len(val))
print("Families:", val["family"].nunique())
print("Label 0:", (val["label"] == 0).sum())
print("Label 1:", (val["label"] == 1).sum())

# ============================================================
# 1. BASIC SEQUENCE LENGTH DISTRIBUTION
# ============================================================

train["length"] = train["sequence"].str.len()
val["length"] = val["sequence"].str.len()

def summarize_numeric(a, b, name):

    ks_stat, ks_p = ks_2samp(
        a,
        b,
    )

    wd = wasserstein_distance(
        a,
        b,
    )

    return {
        "feature": name,
        "train_mean": np.mean(a),
        "val_mean": np.mean(b),
        "train_std": np.std(a),
        "val_std": np.std(b),
        "train_median": np.median(a),
        "val_median": np.median(b),
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "wasserstein": wd,
    }

basic_results = []

basic_results.append(
    summarize_numeric(
        train["length"],
        val["length"],
        "sequence_length",
    )
)

# ============================================================
# 2. PHYSICOCHEMICAL FEATURES
# ============================================================

phys = np.load(
    os.path.join(
        FEATURE_DIR,
        "phys_BioShield.npy",
    ),
    mmap_mode="r",
)

assert phys.shape == (9512, 28)

train_rows = train.index.to_numpy()
val_rows = val.index.to_numpy()

phys_train = np.asarray(
    phys[train_rows],
    dtype=np.float64,
)

phys_val = np.asarray(
    phys[val_rows],
    dtype=np.float64,
)

phys_names = [
    "length",
    "charge",
    "charge_per_length",
    "gravy",
    "pI",
    "aromaticity",
    "instability",
    "hydrophobic_moment",
]

phys_names += [
    f"AAC_{aa}"
    for aa in "ACDEFGHIKLMNPQRSTVWY"
]

print("\n" + "=" * 100)
print("PHYSICOCHEMICAL DISTRIBUTION")
print("=" * 100)

for i, name in enumerate(phys_names):

    result = summarize_numeric(
        phys_train[:, i],
        phys_val[:, i],
        name,
    )

    basic_results.append(result)

    print(
        f"{name:25s} "
        f"TRAIN={result['train_mean']:.4f} "
        f"VAL={result['val_mean']:.4f} "
        f"KS={result['ks_stat']:.4f} "
        f"p={result['ks_p']:.4g}"
    )

basic_df = pd.DataFrame(
    basic_results
)

basic_df.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_train_vs_val_numeric_audit.csv",
    ),
    index=False,
)

# ============================================================
# 3. LABEL-SPECIFIC LENGTH DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("LABEL-SPECIFIC LENGTH DISTRIBUTION")
print("=" * 100)

label_length_rows = []

for label in [0, 1]:

    tr = train.loc[
        train["label"] == label,
        "length",
    ].to_numpy()

    va = val.loc[
        val["label"] == label,
        "length",
    ].to_numpy()

    result = summarize_numeric(
        tr,
        va,
        f"length_label_{label}",
    )

    result["label"] = label

    label_length_rows.append(result)

    print(
        f"Label {label}: "
        f"TRAIN mean={result['train_mean']:.3f}, "
        f"VAL mean={result['val_mean']:.3f}, "
        f"KS={result['ks_stat']:.4f}, "
        f"p={result['ks_p']:.4g}"
    )

label_length_df = pd.DataFrame(
    label_length_rows
)

label_length_df.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_label_specific_length_audit.csv",
    ),
    index=False,
)

# ============================================================
# 4. FAMILY SIZE DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("FAMILY SIZE DISTRIBUTION")
print("=" * 100)

train_family_sizes = (
    train["family"]
    .value_counts()
    .to_numpy()
)

val_family_sizes = (
    val["family"]
    .value_counts()
    .to_numpy()
)

family_result = summarize_numeric(
    train_family_sizes,
    val_family_sizes,
    "family_size",
)

print(
    f"TRAIN mean={family_result['train_mean']:.4f}, "
    f"VAL mean={family_result['val_mean']:.4f}"
)

print(
    f"TRAIN median={family_result['train_median']:.1f}, "
    f"VAL median={family_result['val_median']:.1f}"
)

print(
    f"KS={family_result['ks_stat']:.4f}, "
    f"p={family_result['ks_p']:.4g}"
)

print(
    f"TRAIN max={train_family_sizes.max()}, "
    f"VAL max={val_family_sizes.max()}"
)

# ============================================================
# 5. LARGE FAMILY COUNTS
# ============================================================

print("\nLarge-family counts:")

for cutoff in [2, 3, 5, 10]:

    tr_count = (
        train_family_sizes >= cutoff
    ).sum()

    va_count = (
        val_family_sizes >= cutoff
    ).sum()

    print(
        f"families >= {cutoff}: "
        f"TRAIN={tr_count}, "
        f"VAL={va_count}"
    )

# ============================================================
# 6. CROSS-SPLIT SIMILARITY CHECK
# ============================================================

print("\n" + "=" * 100)
print("TRAIN-VAL SIMILARITY GRAPH AUDIT")
print("=" * 100)

graph = pd.read_csv(
    GRAPH_FILE,
)

print("Graph rows:", len(graph))

# Graph IDs have forms such as:
# V2_2969_label1_label0
#
# Extract the first V2_xxxx identifier.

graph["id1_base"] = (
    graph["seq1"]
    .astype(str)
    .str.extract(
        r"(V2_\d+)",
        expand=False,
    )
)

graph["id2_base"] = (
    graph["seq2"]
    .astype(str)
    .str.extract(
        r"(V2_\d+)",
        expand=False,
    )
)

split_map = d.set_index(
    "v2_id"
)["split_final_BioShield"].to_dict()

graph["split1"] = graph["id1_base"].map(
    split_map
)

graph["split2"] = graph["id2_base"].map(
    split_map
)

tv = graph[
    (
        (
            (graph["split1"] == "train")
            &
            (graph["split2"] == "val")
        )
        |
        (
            (graph["split1"] == "val")
            &
            (graph["split2"] == "train")
        )
    )
].copy()

print("TRAIN-VAL graph pairs:", len(tv))

# ------------------------------------------------------------
# Passing pairs
# ------------------------------------------------------------

tv_pass = tv[
    tv["PASS"].astype(bool)
].copy()

print(
    "TRAIN-VAL PASS pairs:",
    len(tv_pass)
)

if len(tv_pass) == 0:

    print(
        "PASS pairs: ZERO"
    )

else:

    print(
        "\nWARNING: TRAIN-VAL passing similarity pairs found."
    )

    print(
        tv_pass[
            [
                "seq1",
                "seq2",
                "identity",
                "coverage_seq1",
                "coverage_seq2",
            ]
        ].head(20).to_string(
            index=False
        )
    )

# ============================================================
# 7. CROSS-SPLIT MAXIMUM IDENTITY
# ============================================================

print("\n" + "=" * 100)
print("TRAIN-VAL MAXIMUM GRAPH IDENTITY")
print("=" * 100)

if len(tv) > 0:

    tv_sorted = tv.sort_values(
        "identity",
        ascending=False,
    )

    top_tv = tv_sorted.head(20)

    print(
        top_tv[
            [
                "seq1",
                "seq2",
                "identity",
                "coverage_seq1",
                "coverage_seq2",
                "PASS",
            ]
        ].to_string(
            index=False
        )
    )

    top_tv.to_csv(
        os.path.join(
            OUT_DIR,
            "BioShield_train_val_top_identity_pairs.csv",
        ),
        index=False,
    )

    print(
        "\nMaximum TRAIN-VAL identity:",
        f"{tv['identity'].max():.6f}",
    )

else:

    print("No TRAIN-VAL graph pairs found.")

# ============================================================
# 8. IDENTITY DISTRIBUTION
# ============================================================

if len(tv) > 0:

    print("\nTRAIN-VAL identity quantiles:")

    print(
        tv["identity"]
        .quantile(
            [
                0,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
                1.00,
            ]
        )
        .to_string()
    )

# ============================================================
# 9. ESM35 GLOBAL DISTRIBUTION
# ============================================================

print("\n" + "=" * 100)
print("ESM35 EMBEDDING DISTRIBUTION")
print("=" * 100)

esm35 = np.load(
    os.path.join(
        FEATURE_DIR,
        "esm35_BioShield.npy",
    ),
    mmap_mode="r",
)

assert esm35.shape == (9512, 480)

esm_train = np.asarray(
    esm35[train_rows],
    dtype=np.float64,
)

esm_val = np.asarray(
    esm35[val_rows],
    dtype=np.float64,
)

esm_results = []

# Instead of testing all 480 dimensions blindly,
# calculate distribution shift for every dimension
# and report the strongest shifts.

for i in range(esm_train.shape[1]):

    a = esm_train[:, i]
    b = esm_val[:, i]

    ks_stat, ks_p = ks_2samp(
        a,
        b,
    )

    wd = wasserstein_distance(
        a,
        b,
    )

    esm_results.append({
        "dimension": i,
        "train_mean": a.mean(),
        "val_mean": b.mean(),
        "train_std": a.std(),
        "val_std": b.std(),
        "ks_stat": ks_stat,
        "ks_p": ks_p,
        "wasserstein": wd,
    })

esm_df = pd.DataFrame(
    esm_results
)

esm_df = esm_df.sort_values(
    "ks_stat",
    ascending=False,
)

esm_df.to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_esm35_dimension_shift.csv",
    ),
    index=False,
)

print(
    "Dimensions with largest KS shifts:"
)

print(
    esm_df.head(20).to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

print(
    "\nMedian ESM35 KS statistic:",
    f"{esm_df['ks_stat'].median():.6f}",
)

print(
    "Maximum ESM35 KS statistic:",
    f"{esm_df['ks_stat'].max():.6f}",
)

# ============================================================
# 10. FEATURE DISTRIBUTION SUMMARY
# ============================================================

print("\n" + "=" * 100)
print("FEATURE SHIFT SUMMARY")
print("=" * 100)

print("\nPhysicochemical features with largest KS:")
print(
    basic_df.sort_values(
        "ks_stat",
        ascending=False,
    ).head(10)[
        [
            "feature",
            "train_mean",
            "val_mean",
            "ks_stat",
            "ks_p",
        ]
    ].to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}",
    )
)

# ============================================================
# 11. IMPORTANT INTERPRETATION FLAGS
# ============================================================

print("\n" + "=" * 100)
print("AUDIT FLAGS")
print("=" * 100)

max_phys_ks = basic_df[
    basic_df["feature"] != "sequence_length"
]["ks_stat"].max()

length_ks = basic_df.loc[
    basic_df["feature"] == "sequence_length",
    "ks_stat",
].iloc[0]

print(
    f"Maximum physicochemical KS: {max_phys_ks:.4f}"
)

print(
    f"Sequence-length KS:          {length_ks:.4f}"
)

if len(tv_pass) == 0:
    print(
        "TRAIN-VAL PASS similarity:   ZERO"
    )
else:
    print(
        "TRAIN-VAL PASS similarity:   WARNING"
    )

if len(tv) > 0:
    print(
        f"Maximum TRAIN-VAL identity:  "
        f"{tv['identity'].max():.4f}"
    )

print(
    "\nNOTE:"
)
print(
    "Small KS p-values alone do not prove problematic "
    "distribution shift because the sample sizes are large."
)
print(
    "We care primarily about effect size, identity structure, "
    "and whether VAL is unusually easy."
)

# ============================================================
# 12. TEST SAFETY
# ============================================================

print("\n" + "=" * 100)
print("TEST SAFETY")
print("=" * 100)
print("Test sequences loaded: 0")
print("Test predictions generated: 0")
print("Test metrics calculated: 0")
print("TEST REMAINS LOCKED")

# ============================================================
# SAVE SUMMARY
# ============================================================

summary = {
    "train_n": len(train),
    "val_n": len(val),
    "train_families": train["family"].nunique(),
    "val_families": val["family"].nunique(),
    "train_label0": int((train["label"] == 0).sum()),
    "train_label1": int((train["label"] == 1).sum()),
    "val_label0": int((val["label"] == 0).sum()),
    "val_label1": int((val["label"] == 1).sum()),
    "train_val_graph_pairs": len(tv),
    "train_val_pass_pairs": len(tv_pass),
    "train_val_max_identity": (
        float(tv["identity"].max())
        if len(tv) else np.nan
    ),
    "max_phys_ks": float(max_phys_ks),
    "length_ks": float(length_ks),
    "esm35_max_dimension_ks": float(
        esm_df["ks_stat"].max()
    ),
    "esm35_median_dimension_ks": float(
        esm_df["ks_stat"].median()
    ),
}

pd.DataFrame(
    [summary]
).to_csv(
    os.path.join(
        OUT_DIR,
        "BioShield_train_val_audit_summary.csv",
    ),
    index=False,
)

print("\n" + "=" * 100)
print("TRAIN vs VAL AUDIT COMPLETE")
print("=" * 100)

print("\nResults directory:")
print(OUT_DIR)

