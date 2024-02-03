import numpy as np
import torch
from model import Actor, Critic
import torch.optim as optim
import torch.nn.functional as F
import os


class ReplayBuffer():
    def __init__(self, max_size=1000000):
        self.storage = []
        self.max_size = max_size
        self.point = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.point = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, x_prime, u, r, d = [], [], [], [], []

        for i in ind:
            X, X_PRIME, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            x_prime.append(np.array(X_PRIME, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(x_prime), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)


class DDPG:
    def __init__(self,
                 state_dim,
                 action_dim,
                 update_iteration,
                 save_path,
                 max_action,
                 batch_size=1024,
                 gamma=0.99,
                 tau=0.003,
                 device='cpu'):
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._update_iteration = update_iteration
        self._save_path = save_path
        self._max_action = max_action
        self._batch_size = batch_size
        self._gamma = gamma
        self._tau = tau
        self._device = device


        self.actor = Actor(state_dim, action_dim, max_action).to(self._device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(self._device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(self._device)
        self.critic_target = Critic(state_dim, action_dim).to(self._device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = ReplayBuffer()

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.tensor(state.reshape(1, -1), dtype=torch.float32).to(self._device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(self._update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(self._batch_size)
            state = torch.tensor(x, dtype=torch.float32).to(self._device)
            action = torch.tensor(u, dtype=torch.float32).to(self._device)
            next_state = torch.tensor(y, dtype=torch.float32).to(self._device)
            done = torch.tensor(1-d, dtype=torch.float32).to(self._device)
            reward = torch.tensor(r, dtype=torch.float32).to(self._device)

            # Compute Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * self._gamma * target_Q).detach()

            predict_Q = self.critic(state, action)

            # Optimize the critic
            critic_loss = F.mse_loss(predict_Q, target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Optimize the actor
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self._tau * param.data + (1 - self._tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), os.path.join(self._save_path, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(self._save_path, 'critic.pth'))
        print("="*50)
        print("Model has been saved")
        print("="*50)

    def load(self, model_path):
        self.actor.load_state_dict(torch.load(os.path.join(model_path, 'actor.pth')))
        self.critic.load_state_dict(torch.load(os.path.join(model_path, 'critic.pth')))
        print("="*50)
        print("Model has been loaded")
        print("="*50)