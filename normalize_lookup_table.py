import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EPS = 1e-8

INPUT_FILES = {
    "pi_to_pi": os.path.join(
        BASE_DIR, "data", "lookup_table", "pi_to_pi_lookup_results.csv"
    ),
    "pi_to_gpu": os.path.join(
        BASE_DIR, "data", "lookup_table", "pi_to_gpu_lookup_results.csv"
    ),
}

OUTPUT_DIR = os.path.join(BASE_DIR, "data", "lookup_table")

# Metrics
STATE_REWARD_METRICS = [
    "part1_inference_time_s",
    "part2_inference_time_s",
    "system_inference_throughput_imgs_per_s",
]

NETWORK_METRIC = "network_time_s"


def normalize_lookup(df):
    df = df.copy()

    # -------------------------------
    # 1️⃣ Normalize state/reward metrics per (model, bandwidth)
    # -------------------------------
    for (model, bw), idx in df.groupby(
        ["model_name", "bandwidth_mbps"]
    ).groups.items():

        for m in STATE_REWARD_METRICS:
            vals = df.loc[idx, m]
            min_v = vals.min()
            max_v = vals.max()

            if max_v - min_v < EPS:
                df.loc[idx, m] = 0.0
            else:
                df.loc[idx, m] = (vals - min_v) / (max_v - min_v)

    # -------------------------------
    # 2️⃣ Normalize network transfer time per model (ignore bandwidth)
    # -------------------------------
    for model, idx in df.groupby("model_name").groups.items():
        vals = df.loc[idx, NETWORK_METRIC]
        min_v = vals.min()
        max_v = vals.max()

        if max_v - min_v < EPS:
            df.loc[idx, NETWORK_METRIC] = 0.0
        else:
            df.loc[idx, NETWORK_METRIC] = (vals - min_v) / (max_v - min_v)

    return df


# ---------------- RUN NORMALIZATION ---------------- #

os.makedirs(OUTPUT_DIR, exist_ok=True)

for name, in_path in INPUT_FILES.items():
    print(f"Normalizing: {in_path}")

    df = pd.read_csv(in_path)
    df_norm = normalize_lookup(df)

    out_path = os.path.join(
        OUTPUT_DIR, f"{name}_lookup_results_normalized.csv"
    )

    df_norm.to_csv(out_path, index=False)
    print(f"Saved → {out_path}\n")
