import glob
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_model_features(models_dir="data/model_csvs", save=False):
    csv_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".csv")])
    dfs = []

    for file in csv_files:
        model_name = os.path.splitext(file)[0]
        df = pd.read_csv(os.path.join(models_dir, file))
        df["model_name"] = model_name
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    numeric_fields = [
        "FLOPs (G)",
        "Param Memory (MB)",
        "Activation Size (MB)",
        "pi_execution_time"
    ]

    scaler = MinMaxScaler(feature_range=(-1, 1))
    combined_df[numeric_fields] = scaler.fit_transform(combined_df[numeric_fields])

    if save:
        combined_df.to_csv(os.path.join(models_dir, "combined_normalized.csv"), index=False)

        import joblib
        joblib.dump(scaler, os.path.join(models_dir, "system_feature_scaler.pkl"))
        print("✅ Saved scaler for future inference")

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
    Load normalized block metrics.
    Returns: {model_name: blocks_list}

    Each block dict:
    {
        "pi_time": ...,
        "mem_req": ...,
        "activation_size": ...,
        "model": ...
    }
    """
    model_data = {}
    files = sorted(glob.glob(os.path.join(models_dir, "*_block_metrics_batch8_normalized.csv")))

    for filepath in files:
        df = pd.read_csv(filepath)
        model_name = os.path.basename(filepath).replace("_block_metrics_batch8_normalized.csv", "")

        # --- Standardize model names ---
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

        # --- Convert rows to blocks ---
        blocks = []
        for _, row in df.iterrows():
            blocks.append({
                "cpu_time": row["pi_execution_time"],
                "gpu_time": row["gpu_execution_time"],
                "activation_size": row["Activation Size (MB)"],
                "mem_req": row["Param Memory (MB)"],
                "model": row["model_name"]
            })

        model_data[model_name] = blocks

    return model_data



# Example usage
if __name__ == "__main__":
    combined_df = normalize_model_features()
    model_dfs = split_by_model(combined_df, save_dir="data/normalized_model_csvs")

    models = load_models()
    print(f"Found {len(models)} models:")
    for name, blocks in models.items():
        print(f"{name}: {len(blocks)} blocks")
