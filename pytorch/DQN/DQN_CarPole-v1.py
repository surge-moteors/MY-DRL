from collections import namedtuple

import gym
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import BatchSampler, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter

# Hyper-parameters
seed = 1
render = False
num_episodes = 2000
env = gym.make('CartPole-v0').unwrapped
num_state = env.observation_space.shape[0]
num_action = env.action_space.n
torch.manual_seed(seed)
env.seed(seed)

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state'])


class Net(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(num_state, 100)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_value = self.fc2(x)
        return action_value


class DQN:
    # 写在这里定义的时候就是self???
    # capacity = 8000
    # learning_rate = 1e-3
    # batch_size = 256
    # gamma = 0.995

    def __init__(self):
        super(DQN, self).__init__()
        self.capacity = 8000
        self.learning_rate = 1e-3
        self. batch_size = 256
        self.gamma = 0.995
        self.eval_net, self.target_net = Net()
        self.memory = [None] * self.capacity
        self.memory_count = 0
        self.update_count = 0
        self.optimizer = optim.Adam(self.eval_net.parameters(), self.learning_rate)
        self.loss_func = nn.MSELoss()
        self.writer = SummaryWriter('./DQN/logs')  # todo:

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        value = self.eval_net(state)
        action_max_value, index = torch.max(value, 1)
        action = index.item()
        if np.random.rand(1) >= 0.9:
            action = np.random.choice(range(num_action), 1).item()
        return action

    def store_transition(self, transition):
        index = self.memory_count % self.capacity
        self.memory[index] = transition
        self.memory_count += 1   # 这里如果太多的话可能会崩啊
        return self.memory_count >= self.capacity

    def update(self):
        if self.memory_count >= self.capacity:

            # 这里使用了整个数据集
            state = torch.tensor([t.state for t in self.memory]).float()
            action = torch.LongTensor([t.action for t in self.memory]).view(-1, 1).long()
            reward = torch.tensor([t.reward for t in self.memory]).float()
            next_state = torch.tensor([t.next_state for t in self.memory]).float()

            reward = (reward - reward.mean()) / (reward.std() + 1e-7)
            with torch.no_grad():
                target_v = reward + self.gamma * self.target_net(next_state).max(1)[0]

            # todo:
            for index in BatchSampler(SubsetRandomSampler(range(len(self.memory))), batch_size=self.batch_size, drop_last=False):
                eval_value = (self.eval_net(state).gather(1, action))[index]  # 求Q(s, a)
                loss = self.loss_func(target_v[index].unsqueeze(1), eval_value)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar('loss/value_loss', loss, self.update_count)  # todo:
                self.update_count += 1
                if self.update_count % 100 == 0:
                    self.target_net.load_state_dict(self.eval_net.state_dict())
        else:
            print('Memory Buffer is too small')


def main():
    agent = DQN()
    for i_ep in range(num_episodes):
        state = env.reset()
        for t in range(10000):  # todo: the function of 10000
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if render:
                env.render()
            transition = Transition(state, action, reward, next_state)
            agent.store_transition(transition)
            state = next_state
            if done or t >=9999:
                agent.writer.add_scalar('live/finish_step', t+1, global_step=i_ep)
                agent.update()
                if i_ep % 10 == 0:
                    print("episodes {}, step is {} ".format(i_ep, t))
                    break


if __name__ == '__main__':
    main()

