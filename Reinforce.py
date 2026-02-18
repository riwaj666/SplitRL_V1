import random
import torch
import torch.optim as optim
import numpy as np
import os
import csv
import matplotlib.pyplot as plt

from Env import DevicePlacementEnv, pi_to_pi_lookup, pi_to_gpu_lookup
from Model import PolicyNet
from main import load_models


# ---------------------------------------------------
# 📈 Plot utility
# ---------------------------------------------------
def save_reward_plot(rewards, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)

    if not rewards:
        return

    plt.figure(figsize=(10, 5))
    plt.plot(rewards, alpha=0.3, label="Episode Reward")

    if len(rewards) > 100:
        smooth = np.convolve(rewards, np.ones(100) / 100, mode="valid")
        plt.plot(range(99, 99 + len(smooth)), smooth, linewidth=2, label="Moving Avg (100)")

    plt.xlabel("Episode")
    plt.ylabel("Final Reward")
    plt.title("Training Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "reward_train.png"), dpi=300)
    plt.close()


# ---------------------------------------------------
# 🚀 TRAIN POLICY
# ---------------------------------------------------
def train_policy(
    train_models,
    reinforce_env,
    num_episodes=15000,
    lr=1e-3,
    batch_size=5,
    entropy_coeff=0.2,
    epsilon_greedy=0.1,
):

    # ------------------ SEEDS ------------------
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # ------------------ HELPERS ------------------
    def normalize(name):
        return name.strip().lower()

    lookup_table = pi_to_pi_lookup if reinforce_env == "1" else pi_to_gpu_lookup

    # ------------------ DEVICES ------------------
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

    # ------------------ LOAD MODELS ------------------
    raw_models = load_models("data/normalized_model_csvs")
    models = {normalize(k): v for k, v in raw_models.items()}
    train_models = [normalize(m) for m in train_models]
    train_models = [m for m in train_models if m in models and m in lookup_table]

    print("✅ Training on models:", train_models)

    # ------------------ INIT POLICY ------------------
    sample_env = DevicePlacementEnv(
        models[train_models[0]],
        device_list,
        reinforce_env,
        train_models[0],
    )

    policy = PolicyNet(
        state_dim=sample_env.observation_space.shape[0],
        num_devices=sample_env.num_devices,
    )

    optimizer = optim.Adam(policy.parameters(), lr=lr)

    # ------------------ LOGGING ------------------
    os.makedirs("results", exist_ok=True)
    log_path = "results/training_log.csv"

    with open(log_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "episode",
            "model",
            "final_reward",
            "split_point",
            "split_ratio",
        ])

    episode_rewards = []
    batch_memory = []
    baseline = 0.0
    reward_buffer = []

    # ---------------------------------------------------
    # 🧠 TRAIN LOOP
    # ---------------------------------------------------
    for episode in range(1, num_episodes + 1):

        model_name = random.choice(train_models)

        env = DevicePlacementEnv(
            models[model_name],
            device_list,
            reinforce_env,
            model_name,
        )

        state, _ = env.reset()
        done = False

        log_probs = []
        entropies = []
        shaped_reward = 0.0

        while not done:
            state_t = torch.tensor(state, dtype=torch.float32)
            mask_t = torch.tensor(env.get_action_mask(), dtype=torch.float32)

            probs, _ = policy(state_t, mask_t)
            probs = torch.clamp(probs, 1e-8, 1.0)
            probs = probs / probs.sum()

            dist = torch.distributions.Categorical(probs)

            # ---- First block forced ----
            if env.current_block == 0:
                action = torch.tensor(0)
            else:
                action = dist.sample()  # always sample from policy
                log_probs.append(dist.log_prob(action))
                entropies.append(dist.entropy())

            state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            shaped_reward += reward
            shaped_reward -= 0.01 * info.get("transfer_time", 0.0)

        final_reward = float(shaped_reward)
        episode_rewards.append(final_reward)

        if log_probs:
            batch_memory.append({
                "log_probs": torch.stack(log_probs),
                "entropies": torch.stack(entropies),
                "reward": final_reward,
            })

        # ---------------------------------------------------
        # ✅ CORRECT SPLIT METRICS (BLOCK-BASED)
        # ---------------------------------------------------
        # ---------------------------------------------------
        # ✅ CORRECT SPLIT METRICS (BLOCK-BASED)
        # ---------------------------------------------------
        split_point = env.num_blocks  # default: no split

        for b in range(1, len(env.actions_taken)):
            if env.actions_taken[b] != env.actions_taken[b - 1]:
                split_point = b
                break

        split_ratio = split_point / env.num_blocks

        # ---- SAFETY CHECK ----
        assert 1 <= split_point <= env.num_blocks, (
            f"Invalid split point {split_point} "
            f"for {model_name} (blocks={env.num_blocks})"
        )

        # ------------------ SAVE LOG ------------------
        with open(log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                episode,
                model_name,
                final_reward,
                split_point,
            ])

        # ---------------------------------------------------
        # 🔁 UPDATE STEP
        # ---------------------------------------------------
        if episode % batch_size == 0 and batch_memory:

            rewards = [ep["reward"] for ep in batch_memory]
            reward_buffer.extend(rewards)
            reward_buffer = reward_buffer[-100:]

            baseline = 0.9 * baseline + 0.1 * np.mean(rewards)
            std = np.std(reward_buffer) + 1e-8

            batch_loss = 0.0
            for ep in batch_memory:
                advantage = (ep["reward"] - baseline) / std
                advantage = np.clip(advantage, -2.0, 2.0)

                batch_loss += (
                    -ep["log_probs"].mean() * advantage
                    - entropy_coeff * ep["entropies"].mean()
                )

            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            entropy_coeff = max(0.02, entropy_coeff * 0.99)
            batch_memory.clear()

        if episode % 500 == 0:
            avg = np.mean(episode_rewards[-500:])
            print(f"Episode {episode:5d} | Avg Reward: {avg:.4f}")

    # ---------------------------------------------------
    # 💾 SAVE
    # ---------------------------------------------------
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(policy.state_dict(), "checkpoints/policy_net.pth")

    save_reward_plot(episode_rewards)
    print("✅ Training complete. Logs saved to results/training_log.csv")
