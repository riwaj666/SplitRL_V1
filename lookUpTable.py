import pandas as pd

pi_to_pi_df = pd.read_csv("data/lookup_table/normalized_lookup_table_1_pi_to_pi.csv")
pi_to_gpu_df = pd.read_csv("data/lookup_table/normalized_lookup_table_1_pi_to_GPU.csv")

# Convert to nested dictionary: model -> split_point -> info
def df_to_lookup(df):
    lookup = {}
    for _, row in df.iterrows():
        model = row["Model name"]
        split = int(row["Split point"])
        if model not in lookup:
            lookup[model] = {}
        lookup[model][split] = {
            "Partition 1 exec": row["Partition 1 exec time (s)"],
            "Partition 2 exec": row["Partition 2 exec time (s)"],
            "Network Transfer": row["Network Transfer Time (s)"]
        }
    return lookup

pi_to_pi_lookup = df_to_lookup(pi_to_pi_df)
pi_to_gpu_lookup = df_to_lookup(pi_to_gpu_df)
