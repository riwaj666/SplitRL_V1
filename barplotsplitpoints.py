import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ================================
# Load Data
# ================================
file_path = "learned_vs_optimal_split.csv"  # <--- change path if needed

df = pd.read_csv(file_path)

# Ensure column names are correct
df.columns = ["Model", "Optimal_Split", "Learned_Split"]

# Compute Errors
df["Absolute_Error"] = abs(df["Optimal_Split"] - df["Learned_Split"])
df["Relative_Error"] = df["Absolute_Error"] / df["Optimal_Split"].replace(0, 1)

print("\n=== SplitRL Split Point Evaluation ===")
print(df)
print("\nMean Absolute Error:", df["Absolute_Error"].mean())
print("Models predicted exactly correct:",
      (df["Absolute_Error"] == 0).sum(), "/", len(df))

# ================================
# Bar Plot
# ================================
x = np.arange(len(df["Model"]))
width = 0.35

plt.figure(figsize=(10, 6))
plt.bar(x - width/2, df["Optimal_Split"], width, label="Optimal Split", edgecolor="black")
plt.bar(x + width/2, df["Learned_Split"], width, label="Learned Split (SplitRL)", edgecolor="black")

plt.xticks(x, df["Model"], rotation=30)
plt.ylabel("Split Point (Block Index)")
plt.title("Learned vs Optimal Split Points(pi to gpu)")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.5)

# Save figure
plt.tight_layout()
plt.savefig("split_point_comparison.png", dpi=300)
plt.show()

print("\nSaved figure as: split_point_comparison.png")
