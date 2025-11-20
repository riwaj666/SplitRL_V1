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
num_episodes = 15000
lr = 1e-3
gamma = 0.9
batch_size = 5
entropy_coeff = 0.8



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


    final_reward = rewards[-1]
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
        "model_name": model_name,
        "log_probs": torch.stack(log_probs),
        "entropies": torch.stack(entropies),
        "reward": final_reward
    })


    # --- Perform policy update every `batch_size` episodes ---
    if (episode + 1) % batch_size == 0:

        # Initialize per-model baselines & buffers if not exist
        if 'baselines' not in globals():
            baselines = {}
        if 'reward_buffers' not in globals():
            reward_buffers = {}

        batch_loss = 0.0

        # Update per-model rolling rewards and baselines
        for ep_data in batch_memory:
            mname = ep_data.get("model_name", model_name)
            rew = ep_data["reward"]

            if mname not in baselines:
                baselines[mname] = 0.0
            if mname not in reward_buffers:
                reward_buffers[mname] = []

            reward_buffers[mname].append(rew)
            if len(reward_buffers[mname]) > 50:
                reward_buffers[mname].pop(0)

            mean_r = np.mean(reward_buffers[mname])
            std_r = np.std(reward_buffers[mname]) + 1e-8

            baselines[mname] = 0.9 * baselines[mname] + 0.1 * rew
            advantage = (rew - mean_r) / std_r
            advantage_baselined = advantage - baselines[mname]

            ep_loss = -ep_data["log_probs"].sum() * advantage_baselined \
                      - entropy_coeff * ep_data["entropies"].sum()

            batch_loss += ep_loss

        batch_loss /= batch_size

        optimizer.zero_grad()
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
        optimizer.step()

        # Slowly decay entropy coefficient
        entropy_coeff = max(0.05, entropy_coeff * 0.999)
        batch_memory = []


    # --- Progress log ---
    if (episode + 1) % 5 == 0:
        avg_reward = sum(reward_history[-5:]) / min(5, len(reward_history))
        print(f"\n[Episode {episode + 1}] Avg Reward (last 5): {avg_reward:.4f}")



# --- After training ---
print("\n=== Simulation Complete ===")

# --- Plot Reward Trend ---
plt.figure(figsize=(12, 5))

# Raw rewards
plt.plot(reward_history, label="Reward per Episode", alpha=0.4)

# Moving average
window = 50
moving_avg = [np.mean(reward_history[max(0,i-window):i+1]) for i in range(len(reward_history))]
plt.plot(moving_avg, label=f"Moving Avg (window={window})", color='red')

plt.xlabel('Episode')
plt.ylabel('Reward (negative inference time)')
plt.title('Reward vs Episode(pi to pi)')
plt.grid(True)
plt.legend()
plt.show()