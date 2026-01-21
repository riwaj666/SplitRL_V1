import torch
import os
import pandas as pd
from Env import DevicePlacementEnv
from Model import PolicyNet
from main import load_models

# ---------------- CONFIG ----------------
CHECKPOINT_PATH = "checkpoints/policy_net.pth"
MODEL_DIR = "data/normalized_model_csvs"
REINFORCE_ENV = input("Enter which table to look: ")
# --------------------------------------

# ----- Device list (same as training) -----
if REINFORCE_ENV == "1":
    device_list = [
        {"name": "RaspberryPi", "mem_capacity": 4096},
        {"name": "RaspberryPi", "mem_capacity": 4096},
    ]
else:
    device_list = [
        {"name": "RaspberryPi", "mem_capacity": 4096},
        {"name": "GPU", "mem_capacity": 8192},
    ]

# -------- LOAD MODELS ----------
models = load_models(MODEL_DIR)
model_names = list(models.keys())

# -------- INIT POLICY ----------
sample_blocks = models[model_names[0]]
tmp_env = DevicePlacementEnv(sample_blocks, device_list, REINFORCE_ENV, model_names[0])

policy = PolicyNet(
    state_dim=tmp_env.observation_space.shape[0],
    num_devices=tmp_env.num_devices
)

policy.load_state_dict(torch.load(CHECKPOINT_PATH))
policy.eval()

print(f"\nLoaded trained policy from {CHECKPOINT_PATH}")

# -------- EVALUATION (ONE ROLLOUT) ----------
results = []

for model_name in model_names:
    blocks = models[model_name]

    env = DevicePlacementEnv(blocks, device_list, REINFORCE_ENV, model_name)
    state, _ = env.reset()
    done = False

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mask_tensor = torch.tensor(env.get_action_mask(), dtype=torch.float32)

        with torch.no_grad():
            probs, _ = policy(state_tensor, mask_tensor)

        action = torch.argmax(probs).item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # ---- Determine split point (FIRST device change) ----
    split_point = None
    for i in range(1, len(env.actions_taken)):
        if env.actions_taken[i] != env.actions_taken[i - 1]:
            split_point = i
            break
    if split_point is None:
        split_point = len(env.actions_taken)

    results.append({
        "model": model_name,
        "bandwidth_mbps": int(env.bandwidth_mbps),
        "split_point": split_point,
        "reward": round(float(reward), 4),
    })

# -------- SAVE & DISPLAY ----------
df = pd.DataFrame(results)

os.makedirs("data/eval", exist_ok=True)
out_path = "data/eval/eval_single_rollout.csv"
df.to_csv(out_path, index=False)

print("\n=== Single-Rollout Evaluation Results ===\n")
print(df.to_string(index=False))
print(f"\nSaved to → {out_path}")
