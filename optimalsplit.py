import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# ---------------------------------------------------------
# Load tables (replace file names as needed)
# ---------------------------------------------------------
lut = pd.read_csv("data/lookup_table/normalized_lookup_table_1_pi_to_GPU.csv")   # The long table you provided
splits = pd.read_csv("figures/learned_vs_optimal_split.csv")

# Normalize column names
lut.columns = ["Model", "Split", "part1", "part2", "network"]

# ---------------------------------------------------------
# Compute latency & throughput for one (model, split)
# ---------------------------------------------------------
def compute_metrics(model, split):
    row = lut[(lut["Model"] == model) & (lut["Split"] == split)].iloc[0]
    p1, p2, net = row["part1"], row["part2"], row["network"]

    latency = p1 + p2 + net
    throughput = 1 / max(p1, p2, net)
    return latency, throughput

# ---------------------------------------------------------
# Build results
# ---------------------------------------------------------
data = []

for _, row in splits.iterrows():
    model = row["Model name"]
    opt_split = row["optimal_split"]
    learn_split = row["learned_split"]

    opt_lat, opt_tp = compute_metrics(model, opt_split)
    learn_lat, learn_tp = compute_metrics(model, learn_split)

    data.append({
        "Model": model,
        "Optimal Latency": opt_lat,
        "Learned Latency": learn_lat,
        "Optimal Throughput": opt_tp,
        "Learned Throughput": learn_tp
    })

results = pd.DataFrame(data)

# ---------------------------------------------------------
# Plot 1 — Latency
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(results["Model"], results["Optimal Latency"], marker="o", label="Optimal Latency")
plt.plot(results["Model"], results["Learned Latency"], marker="o", label="Learned Latency")
plt.title("Latency Comparison: Optimal splits vs SplitRL(pi to GPU)")
plt.xlabel("Model")
plt.ylabel("Latency (s)")
plt.xticks(rotation=45)
plt.ylim(0, 4)
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------------------------------------------
# Plot 2 — Throughput
# ---------------------------------------------------------
plt.figure(figsize=(10, 5))
plt.plot(results["Model"], results["Optimal Throughput"], marker="o", label="Optimal Throughput")
plt.plot(results["Model"], results["Learned Throughput"], marker="o", label="Learned Throughput")
plt.title("Throughput Comparison: Optimal splits vs SplitRL(pi to GPU)")
plt.xlabel("Model")
plt.ylabel("Throughput (img/s)")
plt.xticks(rotation=45)
plt.ylim(0.5, 3.5)
plt.legend()
plt.tight_layout()
plt.show()
