import argparse
from itertools import count

import gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.distributions import Categorical


class Net(nn.Module):
    def __init__(self,
                 dim_state,
                 dim_action,
                 ):
        super().__init__()
        self.fc1 = nn.Linear(dim_state, 100)
        self.fc2 = nn.Linear(100, dim_action)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        action = self.fc2(x)
        return F.softmax(action, dim=1)


class PG:
    def __init__(self,
                 dim_state,
                 dim_action,
                 lr=1e-4,
                 eps=0.01,
                 ):
        self._policy = Net(dim_state, dim_action)
        self._optimizer = optim.Adam(self._policy.parameters(), lr=lr)
        self._eps = eps

        self._log_probs = []
        self.rewards = []

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self._policy(state)
        # 依概率选取action
        rand = Categorical(probs)
        action = rand.sample()

        self._log_probs.append(rand.log_prob(action))
        return action.item()

    def update(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.rewards[::-1]:
            R = r + args.gamma * R
            rewards.insert(0, R)
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + self._eps)  # 归一化 eps防止分母为0

        for log_prob, reward in zip(self._log_probs, rewards):
            policy_loss.append(-log_prob * reward)
        self._optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self._optimizer.step()

        del self.rewards[:]
        del self._log_probs[:]


def main(env,
         num_state,

         ):
    running_reward = 10
    agent = PG(dim_state=num_state,
               )
    for i_episode in count(10000):
        state = env.reset()
        for t in range(1000):
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            agent.rewards.append(reward)
            if done:
                break
        agent.update()

        # 这里的solve判定有问题。根据任务特性去设定。而且这里与每轮更新的数据量
        # 和轮次有关，与更新程度无关，因此一定是错误的
        running_reward = running_reward * 0.99 + t * 0.01   # todo:这个是做什么东西的

        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}'.format(
                i_episode, t))
        # if running_reward > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {} and "
        #           "the last episode runs to {} time steps!".format(running_reward, t))
        #     break

if __name__ == '__main__':
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
