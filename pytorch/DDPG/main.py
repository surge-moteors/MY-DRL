import torch
import wandb
import gym
import os
from DDPG import DDPG
from itertools import count
import numpy as np
import pygame


def main(mode, save_path='./result', model_path=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = gym.make('Pendulum-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    noise = 0.1
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    epochs = 300
    test_interval = 20
    update_iteration = 200
    log_interval = 20
    wandb.init(project='DDPG',
               config={
                   'epochs': epochs,
               },
               reinit=False,
               )
    wandb.run.name = 'DDPG_final_test'

    agent = DDPG(state_dim=state_dim,
                 action_dim=action_dim,
                 update_iteration=update_iteration,
                 save_path=save_path,
                 max_action=max_action,
                 device=device)

    test_reward = 0
    if mode == 'test':
        if not model_path:
            raise ValueError('Model path is empty!')
        agent.load(model_path)
        for i in range(test_interval):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(np.float32(action))
                if t > 10:
                    test_reward += reward
                env.render()
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                if done:
                    print("Interval {}, the Test_reward is {:0.2f}, the Step is {}".format(i, test_reward, t + 1))
                    test_reward = 0
                    # time.sleep(5)
                    break

                state = next_state

    elif mode == 'train':
        if model_path:
            agent.load_model(model_path)
        total_step = 0
        for i in range(epochs):
            total_reward = 0
            step = 0
            state = env.reset()
            for _ in count():
                action = agent.select_action(state)
                action = (action + np.random.normal(0, noise, size=env.action_space.shape[0])).clip(
                    env.action_space.low, env.action_space.high)
                next_state, reward, done, _ = env.step(action)
                agent.replay_buffer.push((state, next_state, action, reward, np.float_(done)))
                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            wandb.log({
                'epoch': i,
                'total_reward': total_reward
            })
            total_step += step + 1
            print("Total T:{} Epoch: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()

            if (i + 1) % log_interval == 0:
                agent.save()


if __name__ == '__main__':
    # main('train')

    main(mode='test', model_path='./result')
