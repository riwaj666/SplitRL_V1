import gymnasium as gym
from gymnasium import spaces
import numpy as np
from lookUpTable import pi_to_pi_lookup,pi_to_gpu_lookup

MODEL_LIST = ['alexnet', 'inceptionv3', 'mobilenetv2', 'resnet18', 'resnet50', 'vgg16']
MODEL_TO_IDX = {m: i for i, m in enumerate(MODEL_LIST)}

NUM_MODELS = len(MODEL_LIST)


class DevicePlacementEnv(gym.Env):
    """
        Environment for coarse-grained device placement (block-level).
        """
    metadata = {"render_modes": []}

    def __init__(self,blocks, devices,reinforce_env,model_name="unknown",):
        super(DevicePlacementEnv, self).__init__()
        self.blocks = blocks
        self.devices = devices
        self.num_devices = len(devices)
        self.num_blocks = len(blocks)
        self.device_times = [0.0] * self.num_devices
        self.device_loads = [0.0] * self.num_devices
        self.device_mem_used = [0.0] * self.num_devices
        self.prev_device = None
        self.prev_device_onehot = None
        self.current_block = 0
        self.model_name = model_name.strip().lower()
        self.reinforce_env=reinforce_env
        # --- Total execution time (used for fraction_left) ---
        if self.reinforce_env == "1":  # Pi → Pi
            self.total_exec_time = sum(b["cpu_time"] for b in self.blocks)
        else:  # Pi → GPU
            self.total_exec_time = sum(b["gpu_time"] for b in self.blocks)

        self.remaining_exec_time = self.total_exec_time

        self.actions_taken = []
        self.bandwidth_mbps = 939



        # State space (continuous values)
        # [block_flops, remaining_blocks, device_loads, device_mem_used,
        #  activation_size, prev_device_onehot]

        #  self.num_devices = 3 (say, 3 GPUs/TPUs)
        # Current block has FLOPs = 500
        # Remaining blocks = 7
        # Device loads = [0.3, 0.5, 0.1] (fraction of load each device has)
        # Device memory used = [2.0, 4.5, 1.0] (in GB, for instance)
        # Activation size = 1.2 (GB or some normalized number)
        # Previous device one-hot = [0, 1, 0] (last block was assigned to device 1)

        # 1(block_flops)
        # + 1(remaining_blocks)
        # + 3(device_loads)
        # + 3(device_mem_used)
        # + 1(activation_size)
        # + 1(network_transfer_time)
        # + 3(prev_device_onehot)
        # = 13
        # features

        if self.reinforce_env == "1":  # Pi → Pi
            exec_feat_dim = 1
        else:  # Pi → GPU
            exec_feat_dim = 2

        state_dim = (
                exec_feat_dim
                + 1  # remaining blocks
                +1 #fraction_left   ← ADD THIS
                + self.num_devices  # device loads
                + self.num_devices  # device memory
                + 1  # activation size
                + 1  # network transfer time
                + self.num_devices  # prev device one-hot
        )

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(state_dim,), dtype=np.float32)

        # Action space = pick device
        self.action_space = spaces.Discrete(self.num_devices)

        self.reset()

    def _get_state(self):
        block = self.blocks[self.current_block]

        cpu_time = block["cpu_time"]
        gpu_time = block["gpu_time"]
        activation_size = block["activation_size"]

        fraction_left = (
            self.remaining_exec_time / self.total_exec_time
            if self.total_exec_time > 0 else 0.0
        )

        if self.current_block == 0 or self.prev_device is None:
            net_transfer_time = 0.0
        else:
            lookup_table = pi_to_pi_lookup if self.reinforce_env == "1" else pi_to_gpu_lookup
            bw = self.bandwidth_mbps
            split_point = self.current_block

            available_splits = sorted(lookup_table[self.model_name][bw].keys())
            if split_point not in available_splits:
                split_point = available_splits[-1]

            net_transfer_time = lookup_table[self.model_name][bw][split_point]["Network Transfer"]

        # Pi→Pi: only CPU time
        if self.reinforce_env == "1":
            exec_features = np.array([cpu_time], dtype=np.float32)
        else:
            exec_features = np.array([cpu_time, gpu_time], dtype=np.float32)



        state = np.concatenate([
            exec_features,
            np.array([self.num_blocks - self.current_block - 1], dtype=np.float32),
            np.array([fraction_left], dtype=np.float32),
            self.device_loads.astype(np.float32),
            self.device_mem_used.astype(np.float32),
            np.array([activation_size], dtype=np.float32),
            np.array([net_transfer_time], dtype=np.float32),
            self.prev_device_onehot.astype(np.float32)
        ])

        return state

    def step(self, action):
        block = self.blocks[self.current_block]
        model = block["model"]

        # inside step()
        if self.reinforce_env == "1":  # Pi → Pi
            exec_time = block["cpu_time"]
        else:  # Pi → GPU
            if action == 0:  # Pi device
                exec_time = block["cpu_time"]
            else:  # GPU device
                exec_time = block["gpu_time"]

        self.remaining_exec_time -= exec_time

        self.device_loads[action] += exec_time
        self.device_mem_used[action] += block["mem_req"]

        self.prev_device = action
        self.prev_device_onehot = np.zeros(self.num_devices)
        self.prev_device_onehot[action] = 1.0
        self.actions_taken.append(action)



        # ---- Termination logic ----
        self.current_block += 1
        terminated = (self.current_block == self.num_blocks)

        if terminated:
            split_point = None
            for i in range(1, len(self.actions_taken)):
                if self.actions_taken[i] != self.actions_taken[i - 1]:
                    split_point = i
                    break
            if split_point is None:
                split_point = len(self.actions_taken)

            model = self.model_name
            lookup_table = pi_to_pi_lookup if self.reinforce_env == "1" else pi_to_gpu_lookup
            bw = int(self.bandwidth_mbps)

            available_splits = sorted(lookup_table[model][bw].keys())
            if split_point not in available_splits:
                split_point = available_splits[-1]

            info = lookup_table[model][bw][split_point]
            # reward = -max(info["Partition 1 exec"], info["Partition 2 exec"], info["Network Transfer"])
            reward = float(info["Throughput"])

            next_state = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            reward=0.0
            next_state = self._get_state()

        return next_state, reward, terminated, False, {"valid_actions": self.get_action_mask()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_block = 0
        self.device_loads = np.zeros(self.num_devices, dtype=np.float32)
        self.device_mem_used = np.zeros(self.num_devices, dtype=np.float32)
        self.prev_device = None
        self.remaining_exec_time = self.total_exec_time
        self.prev_device_onehot = np.zeros(self.num_devices, dtype=np.float32)
        self.placement_log = []
        self.bandwidth_mbps = 939


        return self._get_state(), {}

    def get_action_mask(self):
        """
        Returns a boolean array of shape (num_devices,)
        True = valid action, False = invalid (memory constraint)
        """
        # If episode terminated, return all False
        if self.current_block >= self.num_blocks:
            return np.zeros(self.num_devices, dtype=bool)

        mask = np.ones(self.num_devices, dtype=bool)
        block = self.blocks[self.current_block]
        for i, device in enumerate(self.devices):
            if self.device_mem_used[i] + block["mem_req"] > device["mem_capacity"]:
                mask[i] = False

        # Sequential constraint: once you move forward, you can't go back
        if self.prev_device is not None:
            for i in range(self.prev_device):
                mask[i] = False  # disable all devices with lower index

        return mask


    def render(self):
        print(f"Block {self.current_block}, Loads: {self.device_loads}, Mem: {self.device_mem_used}")