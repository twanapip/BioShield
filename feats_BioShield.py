import os
import pandas as pd
import numpy as np

from common import phys, embed

DATA = "bioshield_feature_index.csv"
OUTDIR = "BioShield_features"

os.makedirs(OUTDIR, exist_ok=True)

d = pd.read_csv(DATA)

required = [
    "feature_row",
    "v2_id",
    "sequence",
    "label",
    "split_final_BioShield",
]

missing = [x for x in required if x not in d.columns]
if missing:
    raise SystemExit(f"Missing columns: {missing}")

assert len(d) == 9512
assert d["feature_row"].tolist() == list(range(9512))
assert d["v2_id"].nunique() == 9512
assert d["sequence"].nunique() == 9512
assert d["sequence"].notna().all()
assert d["label"].isin([0, 1]).all()
assert d["split_final_BioShield"].isin(["train", "val", "test"]).all()

seqs = d["sequence"].astype(str).tolist()

print("=" * 70)
print("BIOSHIELD FEATURE GENERATION")
print("=" * 70)
print("Sequences:", len(seqs))
print("Train:", (d["split_final_BioShield"] == "train").sum())
print("Val:",   (d["split_final_BioShield"] == "val").sum())
print("Test:",  (d["split_final_BioShield"] == "test").sum())

# ------------------------------------------------------------
# Physicochemical
# ------------------------------------------------------------

print("\nGenerating physicochemical features...")

x = phys(seqs)

assert x.shape == (9512, 28)
assert np.isfinite(x).all()

np.save(f"{OUTDIR}/phys_BioShield.npy", x)

print("phys_BioShield.npy:", x.shape)

# ------------------------------------------------------------
# ESM8
# ------------------------------------------------------------

print("\nGenerating ESM8...")

x = embed(seqs, "esm8", bs=64)

assert x.shape == (9512, 320)
assert np.isfinite(x).all()

np.save(f"{OUTDIR}/esm8_BioShield.npy", x)

print("esm8_BioShield.npy:", x.shape)

# ------------------------------------------------------------
# ESM35
# ------------------------------------------------------------

print("\nGenerating ESM35...")

x = embed(seqs, "esm35", bs=64)

assert x.shape == (9512, 480)
assert np.isfinite(x).all()

np.save(f"{OUTDIR}/esm35_BioShield.npy", x)

print("esm35_BioShield.npy:", x.shape)

# ------------------------------------------------------------
# ESM150
# ------------------------------------------------------------

print("\nGenerating ESM150...")

x = embed(seqs, "esm150", bs=64)

assert x.shape == (9512, 640)
assert np.isfinite(x).all()

np.save(f"{OUTDIR}/esm150_BioShield.npy", x)

print("esm150_BioShield.npy:", x.shape)

# ------------------------------------------------------------
# Manifest
# ------------------------------------------------------------

d.to_csv(
    f"{OUTDIR}/BioShield_feature_manifest.csv",
    index=False
)

print("\n" + "=" * 70)
print("FEATURE GENERATION COMPLETE")
print("=" * 70)

for name in [
    "phys_BioShield.npy",
    "esm8_BioShield.npy",
    "esm35_BioShield.npy",
    "esm150_BioShield.npy",
]:
    path = f"{OUTDIR}/{name}"
    mb = os.path.getsize(path) / (1024 ** 2)
    print(f"{name:20s} {mb:8.2f} MB")

print("\nManifest:")
print("BioShield_features/BioShield_feature_manifest.csv")
