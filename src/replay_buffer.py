from collections import deque
import random
from typing import Tuple

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, terminated: bool):
        self.buffer.append((state, action, reward, next_state, terminated))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        states, actions, rewards, next_states, terminated = zip(*random.sample(self.buffer, batch_size))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(terminated, dtype=torch.float32)
        )

    def clean(self):
        self.buffer.clear()

    def full(self):
        return self.buffer.maxlen == len(self.buffer)

    def __len__(self) -> int:
        return len(self.buffer)
