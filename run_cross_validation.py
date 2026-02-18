import os
import json
import random

from Reinforce import train_policy
from eval_policy import evaluate_policy
from main import load_models

# ---------------------------------------------------
# CONFIG
# ---------------------------------------------------
MODEL_DIR = "data/normalized_model_csvs"
TRAIN_RATIO = 0.8

# Models to exclude entirely
EXCLUDED_MODELS = {"alexnet", "inceptionv3", "mobilenetv2"}

REINFORCE_ENV = input(
    "Enter which table to look (1 = Pi→Pi, 2 = Pi→GPU): "
).strip()



# ---------------------------------------------------
# LOAD MODELS
# ---------------------------------------------------
raw_models = load_models(MODEL_DIR)

# Normalize model names (lowercase keys)
models = {k.lower(): v for k, v in raw_models.items()}

model_names = sorted(models.keys())

# Remove excluded models
available_models = [
    m for m in model_names if m not in EXCLUDED_MODELS
]

print("✅ Models used (excluded alexnet, inceptionv3, mobilenetv2):")
for m in available_models:
    print(f" - {m} ({len(models[m])} blocks)")

# ---------------------------------------------------
# RANDOM TRAIN / TEST SPLIT (80 / 20)
# ---------------------------------------------------
random.shuffle(available_models)

num_models = len(available_models)
num_train = max(1, int(TRAIN_RATIO * num_models))

train_models = available_models[:num_train]
test_models  = available_models[num_train:]

print("\n📊 Data split (80 / 20 random):")
print(f"\nTrain models ({len(train_models)}):")
for m in train_models:
    print(f"  - {m} ({len(models[m])} blocks)")

print(f"\nTest models ({len(test_models)}):")
for m in test_models:
    print(f"  - {m} ({len(models[m])} blocks)")

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
print("\n🚀 Training policy (excluding AlexNet, InceptionV3, MobileNetV2)...")
train_policy(
    train_models=train_models,
    reinforce_env=REINFORCE_ENV,
)

# ---------------------------------------------------
# EVALUATE
# ---------------------------------------------------
print("\n🧪 Evaluating policy on held-out random models...")
df = evaluate_policy(
    test_models=test_models,
    reinforce_env=REINFORCE_ENV,
    fold_id=1
)

output_path = "results/eval_random_excluding_small.csv"
df.to_csv(output_path, index=False)

print("\n✅ Train / test run complete")
print(f"📄 Results saved to {output_path}")
