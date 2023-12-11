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
env = gym.make("CartPole-v1")
learner = DQLearner(env)

learner.train(
    epochs=10000,
)
