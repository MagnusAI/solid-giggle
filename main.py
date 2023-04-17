import gym
import numpy as np

import random


def simulate():
    global epsilon, epsilon_decay

    # Init the environment
    state = env.reset()
    total_reward = 0

    for i in range(MAX_TRY):

        if random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])

        # Perform action and get reward
        next_state, reward, done, _ = env.step(action)
        total_reward += reward

        q_value = q_table[state][action]
        max_q = np.max(q_table[next_state])

        # Update q_table
        q_table[state][action] = (
            1 - learning_rate) * q_value + learning_rate * (reward + gamma * max_q)

        state = next_state

        env.render()

        if done or i == MAX_TRY - 1:
            print('Episode: {}, Total reward: {}, Epsilon: {}'.format(
                i, total_reward, epsilon))
            epsilon *= epsilon_decay
            break

    if epsilon >= 0.005:
        epsilon *= epsilon_decay


if __name__ == '__main__':
    env = gym.make('Lappa-v0')
    MAX_TRY = 1000
    epsilon = 1
    epsilon_decay = 0.999
    learning_rate = 0.1
    gamma = 0.6
    num_box = tuple((env.observation_space.high +
                    np.ones(env.observation_space.shape)).astype(int))
    q_table = np.zeros(num_box + (env.action_space.n,))
    simulate()
