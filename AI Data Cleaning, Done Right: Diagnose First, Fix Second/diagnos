import pandas as pd
import numpy as np
from difflib import SequenceMatcher

df = pd.read_csv("raw_customers.csv")

findings = []

# --- 1. Near-duplicate categories ---
US_STATE_REFERENCE = {
    "ca": "CA", "california": "CA", "calif": "CA",
    "ny": "NY", "new york": "NY", "n y": "NY",
    "tx": "TX", "texas": "TX",
    "wa": "WA", "washington": "WA",
    "fl": "FL", "florida": "FL",
}

def normalize(v):
    return v.lower().replace(".", "").strip()

vals = df["state"].value_counts().index.tolist()
canonical_groups = {}
for v in vals:
    key = normalize(v)
    canon = US_STATE_REFERENCE.get(key)
    # Fuzzy fallback only for values not already resolved by exact reference lookup above.
    # Skip it for short strings: two-letter/three-letter codes collide too easily under
    # similarity scoring (e.g. "WA" vs "CA" can score deceptively high) to trust a fuzzy
    # match instead of an exact hit.
    if canon is None and len(key) >= 4:
        best = max(US_STATE_REFERENCE.values(), key=lambda c: SequenceMatcher(None, key, c.lower()).ratio())
        score = SequenceMatcher(None, key, best.lower()).ratio()
        canon = best if score > 0.6 else None
    if canon:
        canonical_groups.setdefault(canon, []).append(v)

clusters = [group for group in canonical_groups.values() if len(group) > 1]

findings.append({
    "column": "state",
    "issue": "near-duplicate categories",
    "detail": clusters,
    "confidence": "high — same underlying value, different spelling/casing/punctuation",
})

# --- 2. Type mismatch ---
# The column is supposed to hold dates. Some rows instead hold arbitrary
# non-date text (placeholders like "N/A", "pending") — a genuinely
# different kind of value, not the same kind in a different format.
def is_real_date(value):
    try:
        pd.to_datetime(value, format="%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False

mask_non_date = ~df["signup_date"].apply(is_real_date)
pct_non_date = mask_non_date.mean() * 100

findings.append({
    "column": "signup_date",
    "issue": "type mismatch",
    "detail": f"{mask_non_date.sum()} rows ({pct_non_date:.1f}%) hold non-date text instead of a date — "
               f"the column isn't uniformly typed",
    "example_values": sorted(df.loc[mask_non_date, "signup_date"].unique().tolist()),
    "confidence": "high — confirmed by parse failure, not a guess",
})

# --- 3. Distribution anomaly (unit error) ---
q1, q3 = df["session_duration"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
normal = df.loc[df["session_duration"] <= upper_fence, "session_duration"]
above_fence = df.loc[df["session_duration"] > upper_fence, "session_duration"].sort_values()

# IQR outliers alone aren't a finding — a wide or skewed spread can be real
# variance. Only call it a unit error if the outliers are separated from the
# rest by an actual gap (>= 2x): a natural long tail is continuous and has
# no such gap, even if a point near the fence happens to land close to a
# "round" ratio.
cluster_start = None
prev = normal.max()
for v in above_fence:
    if v / prev >= 2.0:
        cluster_start = v
        break
    prev = v
cluster = above_fence[above_fence >= cluster_start] if cluster_start is not None else above_fence.iloc[0:0]

if len(cluster) >= 3:
    normal_median = normal.median()
    ratio_lo, ratio_hi = cluster.min() / normal_median, cluster.max() / normal_median
    findings.append({
        "column": "session_duration",
        "issue": "distribution anomaly suggesting a unit/measurement error",
        "detail": f"{len(cluster)} rows form a cluster {ratio_lo:.1f}x-{ratio_hi:.1f}x the typical value, "
                   f"separated from the rest by a clean gap — a consistent multiple, not random noise",
        "confidence": "medium-high — separated cluster, not just a wide spread",
    })
    unclustered = above_fence[~above_fence.isin(cluster)]
else:
    unclustered = above_fence

if len(unclustered) > 0:
    findings.append({
        "column": "session_duration",
        "issue": "outliers beyond the IQR fence WITHOUT a separating gap (not flagged as a unit error)",
        "detail": f"{len(unclustered)} rows ({sorted(unclustered.tolist())}) sit past the fence but aren't "
                   f"separated from the normal cluster by a real gap — likely genuine tail values, not a unit error",
        "confidence": "low — scattered/continuous, not a distinct cluster",
    })

for f in findings:
    print("=" * 70)
    print(f"COLUMN: {f['column']}")
    print(f"ISSUE:  {f['issue']}")
    for k, v in f.items():
        if k not in ("column", "issue"):
            print(f"  {k}: {v}")
