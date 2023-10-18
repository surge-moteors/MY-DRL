import argparse
from itertools import count

import gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical

parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=1e-2,
                    help='learning rate')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor default: 0.99')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed (default: 1234)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log_interval', type=int, default=10, metavar='N',
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

        self.saved_log_probs = []
        self.rewards = []  # todo:探究这两个是干什么用的

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action = self.fc2(x)
        return F.softmax(action, dim=1)


policy = Policy()
optimizer = optim.Adam(policy.parameters(), lr=args.learning_rate)
eps = np.finfo(np.float32).eps.item()  # 找到大于0的最小的实数


def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    rand = Categorical(probs)  # 按照probs中的概率去生成索引[0, 1]
    action = rand.sample()
    policy.saved_log_probs.append(rand.log_prob(action))  # todo:rand.log_prob(action)求action在rand这个类别分布中的概率的自然对数
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)  # todo: 在list前边插入
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)  # 归一化 eps防止分母为0

    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()

    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01   # todo:这个是做什么东西的
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
