import torch
import numpy as np
import pandas as pd
import os

from Env import DevicePlacementEnv
from Model import PolicyNet
from main import load_models   # ✅ THIS WAS MISSING



def evaluate_policy(test_models, reinforce_env, fold_id, n_rollouts=5):
    # -------- DEVICE LIST --------
    if reinforce_env == "1":
        device_list = [
            {"name": "RaspberryPi", "mem_capacity": 4096},
            {"name": "RaspberryPi", "mem_capacity": 4096},
        ]
    else:
        device_list = [
            {"name": "RaspberryPi", "mem_capacity": 4096},
            {"name": "GPU", "mem_capacity": 8192},
        ]

    all_models = load_models("data/normalized_model_csvs")
    models = {m: all_models[m] for m in test_models}

    # ---- Init policy ----
    sample_blocks = models[test_models[0]]
    tmp_env = DevicePlacementEnv(sample_blocks, device_list, reinforce_env, test_models[0])

    policy = PolicyNet(
        state_dim=tmp_env.observation_space.shape[0],
        num_devices=tmp_env.num_devices
    )

    checkpoint = f"checkpoints/policy_net_fold_{fold_id}.pth"
    policy.load_state_dict(torch.load(checkpoint))
    policy.eval()

    rows = []

    for model_name in test_models:
        rewards = []
        splits = []

        for _ in range(n_rollouts):
            env = DevicePlacementEnv(
                models[model_name],
                device_list,
                reinforce_env,
                model_name
            )
            state, _ = env.reset()
            done = False

            while not done:
                with torch.no_grad():
                    probs, _ = policy(
                        torch.tensor(state, dtype=torch.float32),
                        torch.tensor(env.get_action_mask(), dtype=torch.float32)
                    )

                # deterministic evaluation
                action = torch.argmax(probs).item()
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

            split = next(
                (i for i in range(1, len(env.actions_taken))
                 if env.actions_taken[i] != env.actions_taken[i - 1]),
                len(env.actions_taken) - 1
            )

            rewards.append(float(reward))
            splits.append(split)

        rows.append({
            "fold": fold_id,
            "model": model_name,
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_split": np.mean(splits),
        })

    return pd.DataFrame(rows)


