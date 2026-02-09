import random
import torch
import torch.optim as optim
import numpy as np
import os
import matplotlib.pyplot as plt

from Env import DevicePlacementEnv
from Model import PolicyNet
from main import load_models


# ---------------------------------------------------
# 📈 Plot utility
# ---------------------------------------------------
def save_reward_plot(rewards, fold_id, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    if len(rewards) == 0:
        print("⚠️ No rewards to plot")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.4, label="Episode Reward")

    window = 100
    if len(rewards) >= window:
        smoothed = np.convolve(rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window - 1, window - 1 + len(smoothed)),
            smoothed,
            linewidth=2,
            label="Moving Avg (100)"
        )

    plt.xlabel("Episode")
    plt.ylabel("Final Reward")
    plt.title(f"Reward vs Episode (Fold {fold_id})")
    plt.legend()
    plt.grid(True)

    path = os.path.join(save_dir, f"reward_vs_episode_fold_{fold_id}.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📈 Saved reward plot → {path}")


# ---------------------------------------------------
# 🚀 Train policy
# ---------------------------------------------------
def train_policy(
    train_models,
    reinforce_env,
    fold_id,
    num_episodes=8000,
    lr=1e-3,
    batch_size=5,
    entropy_coeff_init=0.05
):
    # -------- DEVICES --------
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
    model_dir = "data/normalized_model_csvs"
    models = load_models(model_dir)

    # 🔒 SAFETY (critical)
    train_models = [m for m in train_models if m in models]
    if len(train_models) == 0:
        raise ValueError("❌ No valid train models after filtering")

    # -------- INIT POLICY --------
    sample_blocks = models[train_models[0]]
    env = DevicePlacementEnv(sample_blocks, device_list, reinforce_env, train_models[0])

    policy = PolicyNet(
        state_dim=env.observation_space.shape[0],
        num_devices=env.num_devices
    )

    optimizer = optim.Adam(policy.parameters(), lr=lr)
    entropy_coeff = entropy_coeff_init

    global_baseline = 0.0
    reward_buffer = []
    batch_memory = []
    episode_rewards = []

    # -------- TRAIN LOOP --------
    for episode in range(num_episodes):
        model_name = random.choice(train_models)
        env = DevicePlacementEnv(models[model_name], device_list, reinforce_env, model_name)
        state, _ = env.reset()

        done = False
        log_probs, entropies, rewards = [], [], []

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32)
            mask_t = torch.tensor(env.get_action_mask(), dtype=torch.float32)

            probs, _ = policy(state_t, mask_t)
            probs = torch.clamp(probs, 1e-6, 1.0)
            probs = probs / probs.sum()
            dist = torch.distributions.Categorical(probs)

            if env.current_block == 0:
                action = torch.tensor(0)
            else:
                action = dist.sample()
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

            state, reward, terminated, truncated, _ = env.step(action.item())
            rewards.append(reward)
            done = terminated or truncated

        final_reward = rewards[-1]
        # --- after final_reward ---
        num_blocks = len(models[model_name])
        split_ratio = env.current_block / max(1, num_blocks - 1)
        final_reward *= (0.5 + 0.5 * split_ratio)
        final_reward = float(np.clip(final_reward, 1e-6, 1.0))

        episode_rewards.append(final_reward)

        if len(log_probs) > 0:
            batch_memory.append({
                "model": model_name,
                "log_probs": torch.stack(log_probs),
                "entropies": torch.stack(entropies),
                "reward": final_reward
            })

        # -------- UPDATE --------
        if (episode + 1) % batch_size == 0 and batch_memory:
            loss = 0.0

            for ep in batch_memory:
                m = ep["model"]
                r = ep["reward"]

                reward_buffer.append(r)
                reward_buffer = reward_buffer[-100:]

                global_baseline = 0.95 * global_baseline + 0.05 * r
                adv = r - global_baseline
                adv = np.clip(adv, -2.0, 2.0)

                ep_len = ep["log_probs"].shape[0]
                loss += -(ep["log_probs"].sum() / ep_len) * adv \
                        - entropy_coeff * (ep["entropies"].sum() / ep_len)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            entropy_coeff = max(0.005, entropy_coeff * 0.999)
            batch_memory.clear()

        if (episode + 1) % 500 == 0:
            print(f"[Fold {fold_id}] Ep {episode+1} Reward {final_reward:.4f}")

    # -------- SAVE --------
    os.makedirs("checkpoints", exist_ok=True)
    path = f"checkpoints/policy_net_fold_{fold_id}.pth"
    torch.save(policy.state_dict(), path)

    save_reward_plot(episode_rewards, fold_id)
    print(f"✅ Saved policy → {path}")
