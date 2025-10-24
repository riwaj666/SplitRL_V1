import os
import glob
import pandas as pd
from sklearn.preprocessing import StandardScaler

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_flop_model(models_dir="data/model_csvs", field="FLOPs (G)", save=False):
    # Step 1: Collect all CSVs
    csv_files = [f for f in os.listdir(models_dir) if f.endswith(".csv")]
    csv_files.sort()

    dfs = []
    for file in csv_files:
        model_name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(models_dir, file))
        df["model_name"] = model_name
        dfs.append(df)

    # Step 2: Combine
    combined_df = pd.concat(dfs, ignore_index=True)

    # Step 3: Normalize using MinMaxScaler (-1 to 1) and **replace original column**
    if field in combined_df.columns:
        scaler = MinMaxScaler(feature_range=(-1, 1))
        combined_df[[field]] = scaler.fit_transform(combined_df[[field]])
    else:
        print(f"⚠️ Warning: Column '{field}' not found!")

    # Step 4: Optionally save combined CSV
    if save:
        out_path = os.path.join(models_dir, "combined_normalized.csv")
        combined_df.to_csv(out_path, index=False)
        print(f"✅ Saved combined normalized CSV to: {out_path}")

    return combined_df

def split_by_model(combined_df, save_dir="data/normalized_model_csvs"):
    """
    Split combined_df into separate CSVs by model_name and save to save_dir.
    Keeps all columns.
    """
    os.makedirs(save_dir, exist_ok=True)
    model_dfs = {}

    for model_name, group_df in combined_df.groupby("model_name"):
        filename = f"{model_name}_normalized.csv"
        path = os.path.join(save_dir, filename)
        group_df.to_csv(path, index=False)
        model_dfs[model_name] = group_df
        print(f"✅ Saved {model_name} to {path}")

    return model_dfs




def load_models(models_dir="data/normalized_model_csvs"):
    """
    Load all block-metrics CSVs from the directory.
    Returns a dictionary: {model_name: blocks_list}
    Each block is a dict: {"flops": ..., "mem_req": ..., "activation_size": ...}
    """
    model_data = {}
    files = sorted(glob.glob(os.path.join(models_dir, "*_block_metrics_batch8_normalized.csv")))

    for filepath in files:
        df = pd.read_csv(filepath)
        model_name = os.path.basename(filepath).replace("_block_metrics_batch8_normalized.csv", "")
        if model_name.lower().startswith("vgg"):
            model_name = "VGG" + model_name[3:]
        elif model_name.lower().startswith("mobilenetv"):
            model_name = "MobileNetV" + model_name[10:]
        elif model_name.lower().startswith("alexnet"):
            model_name = "AlexNet"
        elif model_name.lower().startswith("inceptionv"):
            model_name = "InceptionV" + model_name[10:]
        elif model_name.lower().startswith("resnet18"):
            model_name = "ResNet18"


        # convert df to blocks list
        blocks = []
        for _, row in df.iterrows():
            blocks.append({
                "flops": float(row["FLOPs (G)"]),
                "mem_req": float(row["Param Memory (MB)"]),
                "activation_size": float(row["Activation Size (MB)"]),
                "model": model_name
            })
        model_data[model_name] = blocks
    return model_data


# Example usage
if __name__ == "__main__":
    # normalize_flop_model()
    # combined_df = normalize_flop_model()
    # model_dfs = split_by_model(combined_df, save_dir="data/normalized_model_csvs")

    models = load_models()
    print(f"Found {len(models)} models:")
    for name, blocks in models.items():
        print(f"{name}: {len(blocks)} blocks")
