import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

pi_to_pi_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "lookup_table", "pi_to_pi_lookup_results_normalized.csv")
)

pi_to_gpu_df = pd.read_csv(
    os.path.join(BASE_DIR, "data", "lookup_table", "pi_to_gpu_lookup_results_normalized.csv")
)


def df_to_lookup(df):
    lookup = {}

    for _, row in df.iterrows():
        model = row["model_name"].strip().lower()
        bw = float(row["bandwidth_mbps"])   # ✅ FLOAT KEY
        split = int(row["split_index"])

        if model not in lookup:
            lookup[model] = {}
        if bw not in lookup[model]:
            lookup[model][bw] = {}

        lookup[model][bw][split] = {
            "Partition 1 exec": float(row["part1_inference_time_s"]),
            "Partition 2 exec": float(row["part2_inference_time_s"]),
            "Network Transfer": float(row["network_time_s"]),
            "Throughput": float(row["system_inference_throughput_imgs_per_s"]),
        }

    return lookup


pi_to_pi_lookup = df_to_lookup(pi_to_pi_df)
pi_to_gpu_lookup = df_to_lookup(pi_to_gpu_df)
