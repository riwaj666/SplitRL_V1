import pandas as pd
import os
from tabulate import tabulate

# ---------------- Config ----------------
INPUT_CSV = "data/lookup_table/pi_to_pi_lookup_results.csv"
OUTPUT_CSV = "data/lookup_table/optimal_points/pi_to_pi_optimal_splits.csv"
OUTPUT_TABLE = "data/lookup_table/optimal_points/pi_to_pi_optimal_splits.txt"

THROUGHPUT_COL = "system_inference_throughput_imgs_per_s"
# ---------------------------------------

df = pd.read_csv(INPUT_CSV)

# Ensure bandwidth is integer (prevents 750.0 / 939.0 bugs)
df["bandwidth_mbps"] = df["bandwidth_mbps"].astype(int)

# Find row with max throughput per (model, bandwidth)
idx = (
    df.groupby(["model_name", "bandwidth_mbps"])[THROUGHPUT_COL]
      .idxmax()
)

optimal_df = (
    df.loc[idx, ["model_name", "bandwidth_mbps", "split_index", THROUGHPUT_COL]]
      .sort_values(["model_name", "bandwidth_mbps"])
      .reset_index(drop=True)
)

# Round for readability
optimal_df[THROUGHPUT_COL] = optimal_df[THROUGHPUT_COL].round(3)

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

# Save CSV (for code / lookup use)
optimal_df.to_csv(OUTPUT_CSV, index=False)

# Create pretty table
table = tabulate(
    optimal_df,
    headers="keys",
    tablefmt="grid",
    showindex=False
)

# Save table to text file
with open(OUTPUT_TABLE, "w") as f:
    f.write(table)

# Print table to console
print("\nOptimal split per model per bandwidth:\n")
print(table)

print(f"\nSaved CSV   → {OUTPUT_CSV}")
print(f"Saved table→ {OUTPUT_TABLE}")
