import torch
import numpy as np
import pandas as pd

from Env import DevicePlacementEnv
from Model import PolicyNet
from main import load_models


def evaluate_policy(test_models, reinforce_env, fold_id=1):

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

    # -------- LOAD MODELS (NORMALIZED) --------
    raw_models = load_models("data/normalized_model_csvs")
    all_models = {k.lower(): v for k, v in raw_models.items()}

    models = {m: all_models[m] for m in test_models}

    # -------- INIT POLICY --------
    sample_blocks = models[test_models[0]]
    tmp_env = DevicePlacementEnv(
        sample_blocks,
        device_list,
        reinforce_env,
        test_models[0]
    )

    policy = PolicyNet(
        state_dim=tmp_env.observation_space.shape[0],
        num_devices=tmp_env.num_devices
    )

    policy.load_state_dict(
        torch.load("checkpoints/policy_net.pth", map_location="cpu")
    )
    policy.eval()

    rows = []

    # -------- EVALUATION --------
    for model_name in test_models:

        env = DevicePlacementEnv(
            models[model_name],
            device_list,
            reinforce_env,
            model_name
        )

        state, _ = env.reset()
        done = False
        shaped_reward = 0.0

        while not done:
            with torch.no_grad():
                probs, _ = policy(
                    torch.tensor(state, dtype=torch.float32),
                    torch.tensor(env.get_action_mask(), dtype=torch.float32)
                )

                # ---- APPLY ACTION MASK ----
                mask = torch.tensor(env.get_action_mask(), dtype=torch.bool)
                probs = probs.clone()
                probs[~mask] = 0.0

                # ---- SAFE STOCHASTIC SELECTION ----
                if probs.sum() == 0:
                    action = torch.argmax(mask.float()).item()
                else:
                    probs = probs / probs.sum()
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample().item()

            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            shaped_reward += reward
            shaped_reward -= 0.01 * info.get("transfer_time", 0.0)

        # -------- SPLIT POINT (LAST SWITCH) --------
        split_point = len(env.actions_taken)
        for i in reversed(range(1, len(env.actions_taken))):
            if env.actions_taken[i] != env.actions_taken[i - 1]:
                split_point = i
                break

        split_ratio = split_point / env.num_blocks

        rows.append({
            "fold": fold_id,
            "model": model_name,
            "final_reward": float(shaped_reward),
            "split_point": split_point,
            "split_ratio": split_ratio
        })

    return pd.DataFrame(rows)
