from flax import linen as nn
import optax
import numpy as np
import gym
import matplotlib.pyplot as plt
import copy
from typing import Any

# hyper-parameter
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.09
EPISILO = 0.9
MEMORY_CAPACITY = 20000
Q_NETWORK_ITERATION = 100

env = gym.make('CartPole-v1')
env = env.unwrapped  # 取消一些限制
NUM_ACTIONS = env.action_space.n
NUM_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample.shape


class Net(nn.Module):

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(feature=50, name='Dense1')(x)
        x = nn.relu(x)
        x = nn.Dense(feature=30, name='Dense2')(x)
        x = nn.relu(x)
        action_prob = nn.Dense(feature=NUM_ACTIONS, name='Dense3')(x)
        return action_prob


class DQN:
    eval_net: Net = Net()
    target_net: Net = Net()
    learn_step_counter: int = 0
    memory_counter: int = 0
    memory: np.ndarray = np.zeros((MEMORY_CAPACITY, NUM_STATES * 2 + 2))
    optimizer: Any = optax.adam(learning_rate=LR)

    def choose_action(self, state):
        pass


dqn=DQN()
print(dqn)


def main():
    pass