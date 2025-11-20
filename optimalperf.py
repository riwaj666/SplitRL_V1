import pandas as pd

# Load the CSV
df = pd.read_csv("data/lookup_table/normalized_lookup_table_1_pi_to_GPU.csv")

# Compute latency and throughput
df["Latency"] = df["Partition 1 exec time (s)"] + df["Partition 2 exec time (s)"] + df["Network Transfer Time (s)"]
df["Throughput"] = 1 / df[["Partition 1 exec time (s)",
                           "Partition 2 exec time (s)",
                           "Network Transfer Time (s)"]].max(axis=1)

# Function to get optimal rows per model
def get_optimal(df):
    results = []

    for model, group in df.groupby("Model name"):
        best_latency_row = group.loc[group["Latency"].idxmin()]
        best_throughput_row = group.loc[group["Throughput"].idxmax()]

        results.append({
            "Model": model,
            "Best Latency Split": int(best_latency_row["Split point"]),
            "Latency": best_latency_row["Latency"],
            "Best Throughput Split": int(best_throughput_row["Split point"]),
            "Throughput": best_throughput_row["Throughput"]
        })

    return pd.DataFrame(results)

optimal_df = get_optimal(df)

# Print results
print(optimal_df)

# Also save to a file
optimal_df.to_csv("optimalperf(pitogpu).csv", index=False)
print("\nSaved to optimal_splits.csv")
