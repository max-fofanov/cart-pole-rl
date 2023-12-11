import random
from collections import deque
from copy import deepcopy
from typing import Tuple

import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.nn import MSELoss
from .model import DQModel
from .replay_buffer import ReplayBuffer


class DQLearner:

    def __init__(self, env, state_size=4, action_size=2):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size

        self.memory = ReplayBuffer(capacity=1000)
        self.sample_size = 32

        self.gamma = 1

        # TODO: create an epsilon scheduler
        self.epsilon = 1.0
        self.epsilon_decay = 0.9995
        self.epsilon_min = 0.1
        self.target_update = 50

        self.live_model = DQModel(self.state_size, self.action_size)
        self.target_model = deepcopy(self.live_model)

    def train(
            self,
            epochs=5000,
    ):
        loss_fn = MSELoss()
        optimizer = Adam(self.live_model.parameters())

        for epoch in tqdm(range(epochs)):

            state, _ = self.env.reset()

            terminated = False

            while not terminated:

                action = self.choose_action(state)
                next_state, reward, terminated, _, _ = self.env.step(action)
                reward = (10 * (2.4 - abs(next_state[0])) + (0.21 - abs(next_state[2]))) if not terminated else 0
                self.memory.push(state, action, reward, next_state, terminated)
                state = next_state

            if self.memory.full():
                optimizer.zero_grad()
                states, actions, rewards, next_states, terminals = self.memory.sample(self.sample_size)
                q_values = self.live_model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                with torch.inference_mode():
                    next_q_values = self.target_model(next_states).max(1)[0]
                target = rewards + self.gamma * next_q_values * (1 - terminals)

                loss = loss_fn(q_values, target)
                loss.backward()
                optimizer.step()

                self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

            if (epoch + 1) % self.target_update == 0:
                test_reward, test_q_value = self.test()
                self.save(epoch, test_reward, test_q_value)
                self.target_model = deepcopy(self.live_model)

    @torch.inference_mode()
    def choose_action(self, state: np.ndarray, inference=False):
        if random.random() < self.epsilon and not inference:
            return np.random.choice(self.action_size)

        output = self.live_model(torch.tensor(state))
        action = torch.argmax(output).item()

        return action

    @torch.inference_mode()
    def test(self, iterations: int = 50) -> Tuple[float, float]:
        rewards = []
        q_values = []

        for i in range(iterations):
            terminated = False
            total_reward = 0
            state, _ = self.env.reset()

            while not terminated:
                output = self.live_model(torch.tensor(state))
                q_values.append(torch.max(output).item())
                action = torch.argmax(output).item()

                next_state, reward, terminated, _, _ = self.env.step(action)
                state = next_state
                total_reward += reward
            rewards.append(total_reward)

        return np.mean(rewards), np.mean(q_values)

    def save(self, epoch: int, reward: float, q_value) -> None:
        torch.save(self.target_model.state_dict(), f"./models/e={epoch}_r={reward}_q={q_value}.pt")
        print(f"Saved a model under ./models/e={epoch}_r={reward}_q={q_value}.pt")

    def load(self, path):
        self.live_model.load_state_dict(torch.load(path))
        self.target_model = deepcopy(self.live_model)






