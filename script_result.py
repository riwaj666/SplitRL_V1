import pandas as pd
import os
from tabulate import tabulate

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "lookup_table",
    "pi_to_pi_lookup_results_normalized.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "lookup_table",
    "by_model_tables"
)

COLUMNS_TO_SAVE = [
    "bandwidth_mbps",
    "split_index",
    "part1_inference_time_s",
    "part2_inference_time_s",
    "network_time_s",
    "system_inference_throughput_imgs_per_s",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_csv(INPUT_FILE)

# Keep only required columns
df = df[["model_name"] + COLUMNS_TO_SAVE]

# Round numeric columns
numeric_cols = df.select_dtypes(include=["float", "int"]).columns
df[numeric_cols] = df[numeric_cols].round(3)

for model_name, model_df in df.groupby("model_name"):
    model_df = model_df.drop(columns=["model_name"]) \
                       .sort_values(["bandwidth_mbps", "split_index"])

    table_str = tabulate(
        model_df,
        headers="keys",
        tablefmt="grid",
        showindex=False
    )

    out_path = os.path.join(
        OUTPUT_DIR,
        f"pi_to_pi_{model_name}_table.md"
    )

    with open(out_path, "w") as f:
        f.write(f"# Model: {model_name}\n\n")
        f.write(table_str)

    print(f"Saved → {out_path}")

print("\n✅ Per-model grid tables created.")
