import os
import torch
import numpy as np
import pandas as pd
import matplotlib

# prevents popup graph windows
matplotlib.use("Agg")

import matplotlib.pyplot as plt

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

    # -------- LOAD MODELS --------
    raw_models = load_models("data/normalized_model_csvs")
    all_models = {k.lower(): v for k, v in raw_models.items()}
    test_models = [m.lower() for m in test_models]

    models = {m: all_models[m] for m in test_models if m in all_models}

    # -------- INIT SAMPLE ENV --------
    sample_blocks = models[test_models[0]]

    tmp_env = DevicePlacementEnv(
        sample_blocks,
        device_list,
        reinforce_env,
        test_models[0]
    )

    # -------- INIT POLICY --------
    policy = PolicyNet(
        state_dim=tmp_env.observation_space.shape[0],
        num_devices=tmp_env.num_devices
    )

    # -------- LOAD TRAINED WEIGHTS --------
    checkpoint_path = f"checkpoints/policy_net_run{fold_id}.pth"

    policy.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )

    policy.eval()

    rows = []
    reward_history = []

    print("\n🧪 Evaluating policy...\n")

    # -------- EVALUATION LOOP --------
    for episode, model_name in enumerate(test_models, 1):

        env = DevicePlacementEnv(
            models[model_name],
            device_list,
            reinforce_env,
            model_name
        )

        state, _ = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            with torch.no_grad():
                state_t = torch.tensor(
                    state,
                    dtype=torch.float32
                )

                mask_t = torch.tensor(
                    env.get_action_mask(),
                    dtype=torch.float32
                )

                probs, _ = policy(state_t, mask_t)

                # -------- APPLY ACTION MASK --------
                mask_bool = torch.tensor(
                    env.get_action_mask(),
                    dtype=torch.bool
                )

                probs = probs.clone()
                probs[~mask_bool] = 0.0

                if probs.sum() == 0:
                    action = torch.argmax(mask_bool.float()).item()
                else:
                    probs = probs / probs.sum()

                    # deterministic evaluation
                    action = torch.argmax(probs).item()

            state, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            total_reward += reward

        # -------- SPLIT POINT --------
        split_point = env.num_blocks

        for i in range(1, len(env.actions_taken)):
            if env.actions_taken[i] != env.actions_taken[i - 1]:
                split_point = i
                break

        split_ratio = split_point / env.num_blocks

        print(
            f"↳ Model: {model_name:15} | "
            f"Bandwidth: {env.bandwidth_mbps:4} Mbps | "
            f"Reward: {total_reward:.4f} | "
            f"Split Point: {split_point}"
        )

        rows.append({
            "fold": fold_id,
            "model": model_name,
            "bandwidth": env.bandwidth_mbps,
            "final_reward": float(total_reward),
            "split_point": split_point,
            "split_ratio": split_ratio
        })

        reward_history.append(total_reward)

    df = pd.DataFrame(rows)

    print("\n📊 Evaluation Summary:")
    print(f"Mean Reward     : {df['final_reward'].mean():.4f}")
    print(f"Std Reward      : {df['final_reward'].std():.4f}")
    print(f"Min Reward      : {df['final_reward'].min():.4f}")
    print(f"Max Reward      : {df['final_reward'].max():.4f}")

    return df

