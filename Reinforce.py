import random
import torch.optim as optim
import torch
from Env import DevicePlacementEnv
from Model import PolicyNet
from main import load_models
import matplotlib.pyplot as plt
import numpy as np
import os

# --------- HYPERPARAMETERS ----------
num_episodes = 200
lr = 1e-4
gamma = 0.9
batch_size = 5

#choose between pi_to_pt and pi_to_gpu env

reinforce_env=input("Enter which table to look:")
if reinforce_env=="1":
    device_list = [
        {"name": "RaspberryPi",  "mem_capacity": 4096},  # example values
        {"name": "RaspberryPi",  "mem_capacity": 4096}
    ]
else:
    device_list = [
        {"name": "RaspberryPi",  "mem_capacity": 4096},  # example values
        {"name": "GPU",  "mem_capacity": 8192}
    ]
# --------- LOAD MODELS ----------
model_dir = "data/normalized_model_csvs"
models = load_models(model_dir)
model_names = list(models.keys())

# --------- SHARED POLICY ----------
# Use one policy for all models
# We'll initialize env with first model just to get state_dim / num_devices
sample_blocks = models[model_names[0]]
env = DevicePlacementEnv(sample_blocks, device_list,reinforce_env,model_names[0])
policy = PolicyNet(state_dim=env.observation_space.shape[0],
                   num_devices=env.num_devices)
optimizer = optim.Adam(policy.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)

baseline = 0.0

reward_history=[]
batch_memory = []

# --------- TRAIN LOOP ----------
gamma = 0.9  # discount factor
entropy_coeff = 0.05  # encourage exploration
batch_size = 5

split_log_path = "data/training_log/split_points_log.txt"

# --- Create directory if missing ---
os.makedirs(os.path.dirname(split_log_path), exist_ok=True)

# --- Initialize split points log file ---
with open(split_log_path, "w") as f:
    f.write("=== Split Points Log ===\n\n")

for episode in range(num_episodes):
    model_name = random.choice(model_names)
    blocks = models[model_name]
    env = DevicePlacementEnv(blocks, device_list, reinforce_env, model_name)
    state, _ = env.reset()
    done = False
    log_probs = []
    entropies = []
    rewards = []

    print(f"\n=== EPISODE {episode + 1} | Model: {model_name} ===")

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mask_tensor = torch.tensor(env.get_action_mask(), dtype=torch.float32)

        probs, _ = policy(state_tensor, mask_tensor)
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

    # --- Device load imbalance penalty ---
    device_counts = [env.actions_taken.count(d) for d in range(env.num_devices)]
    load_penalty = np.std(device_counts)
    final_reward = rewards[-1] - 0.01 * load_penalty
    reward_history.append(final_reward)

    # --- Determine final split point ---
    split_point = None
    for i in range(1, len(env.actions_taken)):
        if env.actions_taken[i] != env.actions_taken[i - 1]:
            split_point = i
            break
    if split_point is None:
        split_point = len(env.actions_taken)

    # --- Log to file: detailed info ---
    with open(split_log_path, "a") as f:
        f.write(f"\n=== Episode {episode + 1} | Model: {model_name} ===\n")
        f.write(f"Split point: {split_point}\n")
        f.write(f"Final reward: {final_reward:.6f}\n")
        f.write("Block → Device mapping:\n")
        for i, d in enumerate(env.actions_taken):
            f.write(f"Block {i} → Device {d}\n")
        f.write("\n")

    # --- Print to terminal: minimal info ---
    print(f"Episode {episode + 1} | Split point: {split_point}")

    # --- Store episode data for batch update ---
    batch_memory.append({
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "reward": final_reward
    })

    # --- Batch update ---
    if (episode + 1) % batch_size == 0:
        # 1) Compute batch reward stats
        rewards_tensor = torch.tensor([ep["reward"] for ep in batch_memory], dtype=torch.float32)
        mean_r = rewards_tensor.mean().item()
        std_r = rewards_tensor.std().item()
        std_r = std_r if std_r > 1e-8 else 1.0  # avoid tiny std -> huge scaling

        # Logging (console)
        print(f"[Batch update] episodes {(episode+1)-batch_size+1}-{episode+1} | mean_reward={mean_r:.4f} std_reward={std_r:.4f}")

        total_loss = 0.0
        # Use a running baseline stored outside the loop (baseline var)
        for ep in batch_memory:
            # normalized reward (simple advantage): (r - mean)/std
            normalized_r = (ep["reward"] - mean_r) / std_r

            # moving baseline (variance reduction)
            baseline = 0.9 * baseline + 0.1 * ep["reward"]
            advantage = ep["reward"] - baseline

            # combine normalized reward and baseline advantage (bounded)
            # keep weighted_r small to prevent huge gradients
            weighted_r = 0.5 * normalized_r + 0.5 * (advantage / (std_r + 1e-8))
            # clamp weighted_r to reasonable range
            weighted_r = float(np.clip(weighted_r, -5.0, 5.0))

            loss = - ep["log_probs"].sum() * weighted_r - entropy_coeff * ep["entropies"].sum()
            total_loss += loss

        optimizer.zero_grad()
        total_loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

        optimizer.step()
        lr_scheduler.step()  # decay lr gently
        batch_memory = []





    # --- Progress log ---
    if (episode + 1) % 5 == 0:
        avg_reward = sum(reward_history[-5:]) / min(5, len(reward_history))
        print(f"\n[Episode {episode + 1}] Avg Reward (last 5): {avg_reward:.4f}")



# --- After training ---
print("\n=== Simulation Complete ===")

# --- Plot Reward Trend ---
plt.figure(figsize=(10, 5))

# Raw rewards
plt.plot(reward_history, label="Reward per Episode", alpha=0.4)

# Moving average
window = 10
moving_avg = [np.mean(reward_history[max(0,i-window):i+1]) for i in range(len(reward_history))]
plt.plot(moving_avg, label=f"Moving Avg (window={window})", color='red')

plt.xlabel('Episode')
plt.ylabel('Reward (negative inference time)')
plt.title('Reward vs Episode')
plt.grid(True)
plt.legend()
plt.show()

# --- 2. Policy Evaluation ---
policy.eval()  # disables dropout, batchnorm updates
num_eval_episodes = 5

# Store results for summary
eval_results = []

for ep in range(num_eval_episodes):
    model_name = random.choice(model_names)
    blocks = models[model_name]
    env = DevicePlacementEnv(blocks, device_list, reinforce_env, model_name)
    state, _ = env.reset()
    done = False
    total_reward = 0

    print(f"\n=== Evaluation Episode {ep + 1} | Model: {model_name} ===")

    while not done:
        state_tensor = torch.tensor(state, dtype=torch.float32)
        mask_tensor = torch.tensor(env.get_action_mask(), dtype=torch.float32)

        with torch.no_grad():
            probs, _ = policy(state_tensor, mask_tensor)

        # --- Force first block on Device 0 ---
        if env.current_block == 0:
            action = 0
        else:
            action = torch.argmax(probs).item()  # greedy for remaining blocks

        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        print(f"Block {env.current_block - 1} → Device {action}")

        done = terminated or truncated

    print(f"Total Reward: {total_reward:.6f}")
    eval_results.append((model_name, total_reward))

# --- Summary Table ---
print("\n=== Evaluation Summary ===")
print(f"{'Model':<15} | {'Total Reward':<15}")
print("-" * 33)
for model_name, total_reward in eval_results:
    print(f"{model_name:<15} | {total_reward:<15.6f}")


