import os
import json
import random
import torch
import numpy as np

from Reinforce import train_policy
from eval_policy import evaluate_policy
from main import load_models

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
MODEL_DIR = "data/normalized_model_csvs"
TRAIN_RATIO = 0.8
NUM_RUNS = 5

EXCLUDED_MODELS = {"alexnet", "inceptionv3", "mobilenetv2"}

REINFORCE_ENV = input(
    "Enter which table to look (1 = Pi→Pi, 2 = Pi→GPU): "
).strip()

# ---------------------------------------------------
# LOAD MODELS (ONCE)
# ---------------------------------------------------
raw_models = load_models(MODEL_DIR)
models = {k.lower(): v for k, v in raw_models.items()}
model_names = sorted(models.keys())

available_models = [
    m for m in model_names if m not in EXCLUDED_MODELS
]

os.makedirs("results", exist_ok=True)
os.makedirs("data/splits", exist_ok=True)

# ---------------------------------------------------
# RUN 5 TIMES
# ---------------------------------------------------
for run_id in range(1, NUM_RUNS + 1):

    seed = 42 + run_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    print(f"\n================ RUN {run_id} ================")

    # --- RANDOM SPLIT ---
    shuffled_models = available_models.copy()
    random.shuffle(shuffled_models)

    num_models = len(shuffled_models)
    num_train = max(1, int(TRAIN_RATIO * num_models))

    train_models = shuffled_models[:num_train]
    test_models  = shuffled_models[num_train:]

    # --- SAVE SPLITS ---
    with open(f"data/splits/train_models_run{run_id}.json", "w") as f:
        json.dump(train_models, f, indent=2)

    with open(f"data/splits/test_models_run{run_id}.json", "w") as f:
        json.dump(test_models, f, indent=2)

    # --- TRAIN ---
    print("🚀 Training policy...")
    train_policy(
        train_models=train_models,
        reinforce_env=REINFORCE_ENV,
        run_id=run_id,
    )

    # --- EVALUATE ---
    print("🧪 Evaluating policy...")
    df = evaluate_policy(
        test_models=test_models,
        reinforce_env=REINFORCE_ENV,
        fold_id=run_id,
    )

    output_path = f"results/eval_random_excluding_small_run{run_id}.csv"
    df.to_csv(output_path, index=False)

    print(f"✅ Run {run_id} complete → {output_path}")

print("\n🎉 All 5 runs completed")
