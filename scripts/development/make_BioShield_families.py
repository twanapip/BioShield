import pandas as pd
import numpy as np
from collections import defaultdict

GRAPH = "bioshield_v2_complete_graph_audit.csv"
INDEX = "bioshield_feature_index.csv"
OUT = "BioShield_similarity_families.csv"

# ============================================================
# LOAD
# ============================================================

g = pd.read_csv(GRAPH)
d = pd.read_csv(INDEX)

assert len(d) == 9512
assert d["v2_id"].nunique() == 9512

# Convert graph IDs such as:
# V2_6454_label0_label0
# back to V2_6454
def base_id(x):
    return str(x).split("_label")[0]

# ============================================================
# UNION-FIND
# ============================================================

ids = d["v2_id"].astype(str).tolist()

parent = {x: x for x in ids}
rank = {x: 0 for x in ids}

def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra = find(a)
    rb = find(b)

    if ra == rb:
        return

    if rank[ra] < rank[rb]:
        parent[ra] = rb
    elif rank[ra] > rank[rb]:
        parent[rb] = ra
    else:
        parent[rb] = ra
        rank[ra] += 1

# ============================================================
# ADD PASSING SIMILARITY EDGES
# ============================================================

passed = g[g["PASS"] == True].copy()

print("Passing edges:", len(passed))

for a, b in zip(passed["seq1"], passed["seq2"]):
    a = base_id(a)
    b = base_id(b)

    if a not in parent:
        raise ValueError(f"Graph ID not in BioShield index: {a}")

    if b not in parent:
        raise ValueError(f"Graph ID not in BioShield index: {b}")

    union(a, b)

# ============================================================
# ASSIGN FAMILY IDS
# ============================================================

roots = {}

for x in ids:
    r = find(x)
    if r not in roots:
        roots[r] = len(roots)

family = {
    x: f"BioShield_FAM_{roots[find(x)]:04d}"
    for x in ids
}

d["family"] = d["v2_id"].map(family)

assert d["family"].notna().all()

# ============================================================
# VALIDATION
# ============================================================

print("\nTotal sequences:", len(d))
print("Unique families:", d["family"].nunique())

sizes = d["family"].value_counts()

print("\nFamily-size distribution:")
print(sizes.describe())

print("\nLargest families:")
print(sizes.head(20))

# ------------------------------------------------------------
# Verify no family crosses BioShield splits
# ------------------------------------------------------------

cross = (
    d.groupby("family")["split_final_BioShield"]
    .nunique()
)

violations = cross[cross > 1]

print("\nCross-split families:", len(violations))

if len(violations):
    print(violations)
    raise SystemExit("ERROR: similarity family crosses BioShield split")

# ------------------------------------------------------------
# Verify each family has one label
# ------------------------------------------------------------

label_counts = (
    d.groupby("family")["label"]
    .nunique()
)

label_conflicts = label_counts[label_counts > 1]

print("Cross-label families:", len(label_conflicts))

if len(label_conflicts):
    print(label_conflicts)
    print(
        "\nNOTE: Cross-label similarity components exist. "
        "They are retained as groups rather than split."
    )

# ============================================================
# SAVE
# ============================================================

cols = [
    "feature_row",
    "v2_id",
    "sequence",
    "label",
    "split_final_BioShield",
    "family",
]

d[cols].to_csv(OUT, index=False)

print("\nSaved:", OUT)
print("Rows:", len(d))
print("Families:", d["family"].nunique())

print("\nFamilies by split:")
print(
    d.groupby("split_final_BioShield")["family"]
     .nunique()
)

print("\nDONE")
