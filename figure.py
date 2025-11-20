import re
import pandas as pd
from scipy.stats import mode

# -------------------------
# Load and parse RL log file
# -------------------------
def parse_rl_log(log_path):
    episodes = []
    current = {}

    with open(log_path, "r") as f:
        for line in f:

            if "Episode" in line and "Model" in line:
                m = re.search(r"Episode\s+(\d+)\s+\|\s+Model:\s+(\S+)", line)
                if m:
                    if current:
                        episodes.append(current)
                    current = {
                        "episode": int(m.group(1)),
                        "model": m.group(2)
                    }

            if "Split point:" in line:
                current["split_point"] = int(re.search(r"Split point:\s+(\d+)", line).group(1))

            if "Final reward:" in line:
                current["reward"] = float(re.search(r"Final reward:\s+([-0-9.]+)", line).group(1))

        if current:
            episodes.append(current)

    return pd.DataFrame(episodes)



# -------------------------
# Compute optimal split point from lookup table
# -------------------------
def compute_optimal_lookup(df_lut):
    df_lut["max_cost"] = df_lut[[
        "Partition 1 exec time (s)",
        "Partition 2 exec time (s)",
        "Network Transfer Time (s)"
    ]].max(axis=1)

    # min max_cost = best split
    optimal = df_lut.loc[df_lut.groupby("Model name")["max_cost"].idxmin()]
    optimal = optimal[["Model name", "Split point"]]
    optimal = optimal.rename(columns={"Split point": "optimal_split"})
    return optimal



# -------------------------
# Compute learned split point (last 20 episodes)
# -------------------------
def compute_learned_split(df_rl, last_n=20):
    learned = (
        df_rl.groupby("model")
            .tail(last_n)
            .groupby("model")["split_point"]
            .apply(lambda x: mode(x, keepdims=True)[0][0])
            .reset_index()
            .rename(columns={"split_point": "learned_split"})
    )
    return learned



# -------------------------
# MAIN
# -------------------------

# 1. Load files
df_rl = parse_rl_log("data/training_log/split_points_log.txt")
df_lut = pd.read_csv("data/lookup_table/normalized_lookup_table_1_pi_to_pi.csv")

# 2. Compute optimal split points from lookup table
df_optimal = compute_optimal_lookup(df_lut)

# 3. Compute learned split points from RL
df_learned = compute_learned_split(df_rl)

# 4. Merge results
df_compare = df_optimal.merge(df_learned, left_on="Model name", right_on="model")
df_compare = df_compare[["Model name", "optimal_split", "learned_split"]]

print("\n=== Learned vs Optimal Split Points ===")
print(df_compare)

# 5. Save
df_compare.to_csv("learned_vs_optimal_split(pi to pi).csv", index=False)
print("\nSaved learned_vs_optimal_split.csv")
