import os
import pandas as pd

DIR = "data/normalized_model_csvs"

for fname in os.listdir(DIR):
    if fname.endswith(".csv"):
        path = os.path.join(DIR, fname)

        df = pd.read_csv(path)

        # 🔥 remove ONLY Unnamed columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

        # 🔁 overwrite same file (in place)
        df.to_csv(path, index=False)

        print(f"Cleaned in-place: {fname}")
