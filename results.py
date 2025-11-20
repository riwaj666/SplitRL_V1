import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Load files
# ---------------------------------------------------
learned_df = pd.read_csv("learned_vs_optimal_split.csv")
lut = pd.read_csv("data/lookup_table/normalized_lookup_table_1_pi_to_pi.csv")


# ---------------------------------------------------
# Helper: compute latency & throughput
# ---------------------------------------------------
def compute_perf(entry):
    p1 = entry["Partition 1 exec time (s)"]
    p2 = entry["Partition 2 exec time (s)"]
    net = entry["Network Transfer Time (s)"]

    latency = p1 + p2 + net
    throughput = 1 / max(p1, p2, net)

    return latency, throughput


# ---------------------------------------------------
# Compute learned split performance
# ---------------------------------------------------
learned_results = []

for _, row in learned_df.iterrows():
    model = row["Model name"]
    split = row["learned_split"]

    match = lut[(lut["Model name"] == model) &
                (lut["Split point"] == split)]

    if match.empty:
        print(f"WARNING: no LUT entry for {model} split {split}")
        continue

    entry = match.iloc[0]
    latency, throughput = compute_perf(entry)

    learned_results.append({
        "Model name": model,
        "Learned Split": split,
        "Learned Latency": latency,
        "Learned Throughput": throughput
    })

learned_df_perf = pd.DataFrame(learned_results)

# ---------------------------------------------------
# Compute optimal performance from LUT
# ---------------------------------------------------
optimal_rows = []

for model, group in lut.groupby("Model name"):
    group = group.copy()

    # compute perf for each split
    group["Latency"] = group.apply(
        lambda r: r["Partition 1 exec time (s)"] +
                  r["Partition 2 exec time (s)"] +
                  r["Network Transfer Time (s)"],
        axis=1
    )
    group["Throughput"] = group.apply(
        lambda r: 1 / max(
            r["Partition 1 exec time (s)"],
            r["Partition 2 exec time (s)"],
            r["Network Transfer Time (s)"]
        ),
        axis=1
    )

    # find minimum latency row
    best_latency_row = group.loc[group["Latency"].idxmin()]

    optimal_rows.append({
        "Model name": model,
        "Optimal Split": int(best_latency_row["Split point"]),
        "Optimal Latency": best_latency_row["Latency"],
        "Optimal Throughput": best_latency_row["Throughput"]
    })

optimal_df = pd.DataFrame(optimal_rows)

# ---------------------------------------------------
# Merge learned vs optimal results
# ---------------------------------------------------
merged = learned_df_perf.merge(optimal_df, on="Model name")
merged.to_csv("learned_vs_optimal_latency_throughput.csv", index=False)

print("\n=== Saved: learned_vs_optimal_latency_throughput.csv ===\n")
print(merged)

# ---------------------------------------------------
# Plot both learned and optimal
# ---------------------------------------------------
plt.figure(figsize=(12, 6))

# Latency
plt.plot(merged["Model name"], merged["Learned Latency"],
         marker="o", label="Learned Latency")

plt.plot(merged["Model name"], merged["Optimal Latency"],
         marker="o", label="Optimal Latency")

# Throughput
plt.plot(merged["Model name"], merged["Learned Throughput"],
         marker="s", label="Learned Throughput")

plt.plot(merged["Model name"], merged["Optimal Throughput"],
         marker="s", label="Optimal Throughput")

plt.xlabel("Model")
plt.ylabel("Time(s)")
plt.title("Learned vs Optimal Latency & Throughput (pi to pi)")
plt.legend()
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("learned_vs_optimal_latency_throughput_plot_pi_to_pi.png", dpi=300)
plt.show()
