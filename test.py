import random
import numpy as np
import torch
import gym
from src import DQLearner

# set everything random for reproducibility
torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

# create an environment
env = gym.make("CartPole-v1", render_mode="human")
learner = DQLearner(env)
learner.load("./models/e=3649_r=701.04_q=404.06442136582086.pt")


state, _ = env.reset()
env.render()

total_reward = 0
terminated = False

while not terminated:
    action = learner.choose_action(state, inference=True)
    state, reward, terminated, _, _ = env.step(action)
    total_reward += reward

print(total_reward)
env.close()
