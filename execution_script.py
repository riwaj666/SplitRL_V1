import pandas as pd
import os

profile_folder = "data/model_csvs"
exec_folder = "data/blockwise_execution_time"

exec_files = os.listdir(exec_folder)

for profile_file in os.listdir(profile_folder):

    if not profile_file.endswith(".csv"):
        continue

    base_name = profile_file.split("_block_metrics")[0].lower()
    profile_path = os.path.join(profile_folder, profile_file)

    matching_exec = None
    for f in exec_files:
        if base_name in f.lower():
            matching_exec = f
            break

    if matching_exec is None:
        print(f"❌ No execution file found for {base_name}")
        continue

    exec_path = os.path.join(exec_folder, matching_exec)

    print(f"🔄 Merging {profile_file}  ⟷  {matching_exec}")

    prof_df = pd.read_csv(profile_path)
    exec_df = pd.read_csv(exec_path)

    # Normalize merge column
    prof_df = prof_df.rename(columns={"Block": "block_number"})

    # 🔥 DROP execution-time columns if they already exist
    overlapping_cols = set(prof_df.columns) & set(exec_df.columns)
    overlapping_cols.discard("block_number")

    if overlapping_cols:
        print(f"♻️ Replacing existing columns: {sorted(overlapping_cols)}")
        prof_df = prof_df.drop(columns=list(overlapping_cols))

    # Merge cleanly
    merged = pd.merge(prof_df, exec_df, on="block_number", how="left")

    merged.to_csv(profile_path, index=False)
    print(f"✅ Updated {profile_file}")

print("\n🎉 Done — all possible files merged")
