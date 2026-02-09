import os
import json
from sklearn.model_selection import KFold

from Reinforce import train_policy
from eval_policy import evaluate_policy
from main import load_models

MODEL_DIR = "data/normalized_model_csvs"
N_FOLDS = 5
REINFORCE_ENV = input("Enter which table to look: ")

# ---------------------------------------------------
# ✅ SINGLE SOURCE OF TRUTH
# ---------------------------------------------------
models = load_models(MODEL_DIR)
model_names = sorted(models.keys())

print("✅ Models loaded:")
for m in model_names:
    print(" -", m)

# ---------------------------------------------------
# K-FOLD SPLIT
# ---------------------------------------------------
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

os.makedirs("data/splits", exist_ok=True)
os.makedirs("results", exist_ok=True)

all_results = []

for fold_id, (train_idx, test_idx) in enumerate(kf.split(model_names), start=1):
    print(f"\n===== FOLD {fold_id}/{N_FOLDS} =====")

    train_models = [model_names[i] for i in train_idx]
    test_models  = [model_names[i] for i in test_idx]

    print("Train models:", train_models)
    print("Test models :", test_models)

    # ---- SAVE SPLITS ----
    with open(f"data/splits/train_fold_{fold_id}.json", "w") as f:
        json.dump(train_models, f, indent=2)

    with open(f"data/splits/test_fold_{fold_id}.json", "w") as f:
        json.dump(test_models, f, indent=2)

    # ---- TRAIN ----
    train_policy(
        train_models=train_models,
        reinforce_env=REINFORCE_ENV,
        fold_id=fold_id
    )

    # ---- EVALUATE ----
    df = evaluate_policy(
        test_models=test_models,
        reinforce_env=REINFORCE_ENV,
        fold_id=fold_id
    )

    df.to_csv(f"results/eval_fold_{fold_id}.csv", index=False)
    all_results.append(df)

print("\n✅ Cross-validation complete")
