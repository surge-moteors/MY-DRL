import numpy as np
from env import MyGridWorld

DISCOUNT_FACTOR = 1


class Agent:
    def __init__(self, env):
        self.env = env
        self.V = np.zeros(self.env.state_dim)

    def next_best_action(self, s, V):
        action_values = np.zeros(self.env.action_dim)
        for a in range(self.env.action_dim):
            for prob, next_state, reward, done in self.env.P[s][a]:
                action_values[a] += prob * (reward + DISCOUNT_FACTOR * V[next_state])
        return np.argmax(action_values), np.max(action_values)

    def optimize(self):
        THETA = 0.0001
        delta = float('inf')
        round_num = 0
        while delta > THETA:
            delta = 0
            print('\nValue Iteration: Round ' + str(round_num))
            print(np.reshape(self.V, self.env.shape))
            for s in range(self.env.state_dim):
                best_action, best_action_value = self.next_best_action(s, self.V)
                delta = max(delta, np.abs(best_action_value - self.V[s]))
                print(best_action_value)
                print(123)
                print(delta)
                self.V[s] = best_action_value
            print(delta)
            round_num += 1

        policy = np.zeros(self.env.state_dim)
        for s in range(self.env.state_dim):
            best_action, best_action_value = self.next_best_action(s, self.V)
            policy[s] = best_action

        return policy


if __name__ == '__main__':
    env = MyGridWorld()
    agent = Agent(env)
    policy = agent.optimize()
    print('\nBest Policy')
    print(np.reshape([env.get_action_name(entry) for entry in policy], env.shape))
