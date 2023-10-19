import random
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss
from .model import DQModel
from .replay_buffer import ReplayBuffer


class DQLearner:

    def __init__(self, env):
        self.env = env
        self.num_actions = 2

        self.live_model = DQModel()
        self.target_model = deepcopy(self.live_model)

    def train(
            self,
            epochs=1000,
            gamma=1,
            epsilon=0.1,
            buffer_size=300,
            batch_size=100,
            epsilon_decay_rate=0.999,
            gamma_increase_rate=1.01,
            target_update_rate=100
    ):
        loss_fn = MSELoss()
        optimizer = Adam(self.live_model.parameters())
        replay_buffer = ReplayBuffer(buffer_size)

        for epoch in tqdm(range(epochs)):

            while True:
                state, _ = self.env.reset()

                terminated = False

                while not terminated:

                    action = self.choose_action(state, epoch, epsilon)
                    next_state, reward, terminated, _, _ = self.env.step(action)
                    reward -= abs(next_state[0] / 2.4)
                    replay_buffer.push(state, action, reward, next_state, terminated)

                    state = next_state

                if len(replay_buffer) >= buffer_size:
                    break

            states, actions, rewards, next_states, terminals = replay_buffer.sample(batch_size)

            optimizer.zero_grad()
            q_values = self.live_model(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
            with torch.inference_mode():
                next_q_values = self.target_model(next_states).max(1)[0]
            target = rewards + gamma * next_q_values * (1 - terminals)

            loss = loss_fn(q_values, target)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % target_update_rate == 0:
                self.target_model = deepcopy(self.live_model)
                test_reward = self.test()
                self.save(epoch, test_reward)

            epsilon = epsilon * epsilon_decay_rate
            gamma = min(gamma * gamma_increase_rate, 1)

    @torch.inference_mode()
    def choose_action(self, state: np.ndarray, epoch: int = 1, epsilon: float = 0):
        if epoch == 0 or random.random() < epsilon:
            return np.random.choice(self.num_actions)

        output = self.live_model(torch.tensor(state))
        action = torch.argmax(output).item()

        return action

    @torch.inference_mode()
    def test(self, iterations: int = 10) -> float:
        rewards = []

        for i in range(iterations):
            terminated = False
            total_reward = 0
            state, _ = self.env.reset()

            while not terminated:
                output = self.live_model(torch.tensor(state))
                action = torch.argmax(output).item()

                next_state, reward, terminated, _, _ = self.env.step(action)
                state = next_state
                total_reward += reward
            rewards.append(total_reward)

        return np.median(rewards)

    def save(self, epoch: int, reward: float) -> None:
        torch.save(self.target_model.state_dict(), f"./models/e={epoch}_r={reward}_.pt")
        print(f"Saved a model under ./models/e={epoch}_r={reward}_.pt")

    def load(self, path):
        self.live_model.load_state_dict(torch.load(path))
        self.target_model = deepcopy(self.live_model)






