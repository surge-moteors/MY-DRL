import argparse

import gym
import torch
from torch import nn
import torch.nn.functional as F
parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor default: 0.99')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()
env = gym.make('CartPole-v1')
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
env.seed(args.seed)
torch.manual_seed(args.seed)


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, num_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action = self.fc2(x)
        return F.softmax(action, dim=1)

Policy = Policy()
