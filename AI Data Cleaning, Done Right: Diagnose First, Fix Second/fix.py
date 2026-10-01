import pandas as pd

df = pd.read_csv("raw_customers.csv")

# repr() instead of plain printing: makes invisible differences like a
# trailing space visible before the fix runs, not just after.
print("Raw state values before fixing:")
for value, count in df["state"].value_counts().items():
    print(f"  {value!r}: {count}")
print()

# --- Fix 1: standardize state values to canonical postal codes ---
state_map = {
    "ca": "CA", "california": "CA", "calif": "CA",
    "ny": "NY", "new york": "NY", "n y": "NY",
    "tx": "TX", "texas": "TX",
    "wa": "WA", "washington": "WA",
    "fl": "FL", "florida": "FL",
}
df["state"] = (
    df["state"]
    .str.lower()
    .str.replace(".", "", regex=False)
    .str.strip()
    .map(state_map)
)

# --- Fix 2: separate real dates from non-date placeholder text ---
# Coerce to a proper datetime dtype; anything that wasn't a real date
# becomes a clean null (NaT), and we flag which rows those were instead
# of silently losing the distinction.
df["signup_date_was_placeholder"] = ~df["signup_date"].str.match(r"^\d{4}-\d{2}-\d{2}$")
df["signup_date"] = pd.to_datetime(df["signup_date"], format="%Y-%m-%d", errors="coerce")

# --- Fix 3: flag (not silently drop) the session_duration outlier cluster ---
# Match the diagnostic: only flag the gap-separated cluster, not every point
# past the IQR fence. A value just past the fence with no real gap from the
# normal range (like 32.2) is a genuine tail value, not the mechanical error
# this flag is meant to call out — flagging it the same way would be a false
# positive the diagnostic already knows to avoid.
q1, q3 = df["session_duration"].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr
normal_max = df.loc[df["session_duration"] <= upper_fence, "session_duration"].max()
above_fence = df.loc[df["session_duration"] > upper_fence, "session_duration"].sort_values()
prev, cluster_start = normal_max, None
for v in above_fence:
    if v / prev >= 2.0:
        cluster_start = v
        break
    prev = v
df["session_duration_flag"] = df["session_duration"] >= cluster_start if cluster_start is not None else False

print(df["state"].value_counts())
print()
print("Any unmapped states:", df["state"].isna().sum())
print()
print("signup_date dtype:", df["signup_date"].dtype)
print("Flagged as placeholder:", df["signup_date_was_placeholder"].sum())
print("Rows where signup_date is now NaT:", df["signup_date"].isna().sum())
print()
print("Flagged session_duration rows:", df["session_duration_flag"].sum())
print(df.head(12))

# Write to a new file — never overwrite the input. Review the result before
# treating it as the "real" data; nothing here deletes or replaces raw_customers.csv.
out_path = "raw_customers_fixed.csv"
df.to_csv(out_path, index=False)
print(f"\nWrote {out_path} ({len(df)} rows). raw_customers.csv was not modified.")
