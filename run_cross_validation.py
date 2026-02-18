import os
import json
import random

from Reinforce import train_policy
from eval_policy import evaluate_policy
from main import load_models

MODEL_DIR = "data/normalized_model_csvs"
TRAIN_RATIO = 0.8

REINFORCE_ENV = input("Enter which table to look: ")

# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
models = load_models(MODEL_DIR)
model_names = sorted(models.keys())

print("✅ Models loaded:")
for m in model_names:
    print(" -", m)

# ---------------------------------------------------
# TRAIN / TEST SPLIT (80 / 20)
# ---------------------------------------------------
# TRAIN / TEST = ALL MODELS
train_models = model_names
test_models  = model_names

print("\n📊 Data split:")
print(f"Train models ({len(train_models)}):", train_models)
print(f"Test models  ({len(test_models)}):", test_models)



# ---------------------------------------------------
# SAVE SPLITS
# ---------------------------------------------------
os.makedirs("data/splits", exist_ok=True)
os.makedirs("results", exist_ok=True)

with open("data/splits/train_models.json", "w") as f:
    json.dump(train_models, f, indent=2)

with open("data/splits/test_models.json", "w") as f:
    json.dump(test_models, f, indent=2)

# ---------------------------------------------------
# TRAIN
# ---------------------------------------------------
print("\n🚀 Training policy on 80% of models...")
train_policy(
    train_models=train_models,
    reinforce_env=REINFORCE_ENV,
)

# ---------------------------------------------------
# EVALUATE
# ---------------------------------------------------
print("\n🧪 Evaluating policy on 20% held-out models...")
df = evaluate_policy(
    test_models=test_models,
    reinforce_env=REINFORCE_ENV,
    fold_id=1
)

df.to_csv("results/eval_80_20.csv", index=False)

print("\n✅ Train / test run complete")
print("📄 Results saved to results/eval_80_20.csv")
