import pandas as pd
import os

# -----------------------------
# Configuration
# -----------------------------
LOOKUP_FILE = "data/lookup_table/normalized_lookup_table_1_pi_to_GPU.csv"
BLOCKS_DIR = "data/normalized_model_csvs"
OUTPUT_DIR = "data/normalized_model_csvs_pi_to_GPU"

BLOCK_SUFFIX = "_block_metrics_batch8_normalized.csv"

# -----------------------------
# Create output directory
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Load lookup table
# -----------------------------
lookup = pd.read_csv(LOOKUP_FILE)
lookup["Model name"] = lookup["Model name"].str.lower()

# -----------------------------
# Process per model (NOT per split)
# -----------------------------
for model in lookup["Model name"].unique():

    block_file = os.path.join(
        BLOCKS_DIR,
        f"{model}{BLOCK_SUFFIX}"
    )

    if not os.path.exists(block_file):
        print(f"[SKIP] Missing block file: {block_file}")
        continue

    # Load model block CSV
    df = pd.read_csv(block_file)

    # Initialize network transfer column
    df["Network Transfer Time (s)"] = 0.0

    # Get lookup rows for this model
    model_lookup = lookup[lookup["Model name"] == model]

    # Assign network transfer time at split blocks
    for _, row in model_lookup.iterrows():
        split_point = int(row["Split point"])
        net_time = float(row["Network Transfer Time (s)"])

        df.loc[df["Block"] == split_point, "Network Transfer Time (s)"] = net_time

    # Save ONE file per model
    output_file = os.path.join(
        OUTPUT_DIR,
        f"{model}_block_metrics_with_network.csv"
    )

    df.to_csv(output_file, index=False)
    print(f"[OK] Saved: {output_file}")

print("\nAll models processed successfully.")
