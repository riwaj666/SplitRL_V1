import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

# Define paths
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.join(script_dir, "data", "lookup_table")

# Columns to normalize
cols_to_normalize = [
    "Partition 1 exec time (s)",
    "Partition 2 exec time (s)",
    "Network Transfer Time (s)"
]

for file in os.listdir(input_dir):
    if file.endswith(".csv"):
        file_path = os.path.join(input_dir, file)
        df = pd.read_csv(file_path)

        # Check if required columns exist
        missing_cols = [col for col in cols_to_normalize if col not in df.columns]
        if missing_cols:
            print(f"⚠️ Skipping {file} — missing columns: {missing_cols}")
            continue

        df_norm = df.copy()

        # Normalize per model using MinMaxScaler
        for model in df["Model name"].unique():
            mask = df["Model name"] == model

            scaler = MinMaxScaler(feature_range=(0, 1))
            df_norm.loc[mask, cols_to_normalize] = scaler.fit_transform(
                df.loc[mask, cols_to_normalize]
            )

        # Save with prefix
        output_file = f"normalized_{file}"
        output_path = os.path.join(input_dir, output_file)
        df_norm.to_csv(output_path, index=False)

        print(f"✅ Saved normalized file: {output_path}")
