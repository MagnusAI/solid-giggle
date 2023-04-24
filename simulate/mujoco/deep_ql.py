# Deep Q-Learning implementation for the Lappa robot.

from api import *
from dqn import DQN
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys

# Define the action space
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward', 'stop_a_rotation', 'stop_b_rotation']

# Define the state space
state_space = list(itertools.product(
    [False, True],
    [False, True],
    [False, True],
    [-1, 0, 5, 10, 15, 20, 25, 30],
    [-1, 0, 5, 10, 15, 20, 25, 30],
    [False, True],
    [False, True]
))

# Define state_dimensions
state_dimensions = len(state_space)
action_dimensions = len(action_space)

device = torch.device("cuda" if torch.cuda.is_available()
                      else "cpu")  # Use GPU if available

input_dimensions = len(state_space[0])
output_dimensions = len(action_space)

q_network = DQN(input_dimensions, output_dimensions).to(device)
target_network = DQN(input_dimensions, output_dimensions).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
    return next_state


def get_reward(state, next_state):
    goal_condition = (next_state[0] and next_state[1]
                      and next_state[5] and next_state[6])
    reset_condition = (
        not next_state[0] and not next_state[1] and (state[0] or state[1])) or (not next_state[0] and not next_state[1] and next_state[3] > state[3] and next_state[4] > state[4])

    if goal_condition:
        return 100
    elif reset_condition:
        return -100
    else:
        if (not next_state[0] and not next_state[1]):
            return -5

        height_reward = 0
        if next_state[5] and not state[5]:
            height_reward += 10
        elif not next_state[5] and state[5]:
            height_reward -= 2
        if next_state[6] and not state[6]:
            height_reward += 10
        elif not next_state[6] and state[6]:
            height_reward -= 2

        fix_reward = 0
        if (next_state[5]):
            if next_state[0] and not state[0]:
                fix_reward += 1
            elif not next_state[0] and state[0]:
                fix_reward -= 1
        if (next_state[6]):        
            if next_state[1] and not state[1]:
                fix_reward += 1
            elif not next_state[1] and state[1]:
                fix_reward -= 1

        acc = height_reward + fix_reward

        return acc - 1


# Define the hyperparameters
learning_rate = 0.001
gamma = 0.9
epsilon = 0.9  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.999
target_update = 10
steps_done = 0
robot = None
state = (False, False, False, 5, 5, False, False)
episodes = 1000
stale_count = 0
stale_limit = 1000


def controller(model, data):
    global robot, state, q_network, target_network, optimizer, epsilon, episodes, steps_done, stale_count, stale_limit

    if (robot is None):
        robot = LappaApi(data)
        try:
            q_network.load_state_dict(torch.load(
                'q_network.pth', map_location=device))
            print('Loaded network from disk')
        except:
            pass
        return
    else:
        robot.update_data(data)

    done = False

    if (episodes <= 0):
        done = True

    if (not done):
        state_tensor = torch.tensor(
            [state], dtype=torch.float32, device=device)

        if np.random.rand() < epsilon:
            action_idx = np.random.choice(action_dimensions)
        else:
            with torch.no_grad():
                action_idx = torch.argmax(q_network(state_tensor)).item()

        action = action_space[action_idx]
        next_state = perform_action(robot, action)

        # increment the stale count if the state is the same
        if (state == next_state):
            stale_count += 1

        next_state_tensor = torch.tensor(
            [next_state], dtype=torch.float32, device=device)

        reward = get_reward(state, next_state)

        if (stale_count > stale_limit):
            reward = -100

        reward_tensor = torch.tensor(
            [reward], dtype=torch.float32, device=device).squeeze()

        current_q_value = q_network(state_tensor)[0][action_idx]
        with torch.no_grad():
            next_q_value = torch.max(target_network(next_state_tensor))
        expected_q_value = reward_tensor + gamma * next_q_value

        loss = loss_fn(current_q_value, expected_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epsilon *= epsilon_decay
        state = next_state

        if (reward == 100 or reward == -100):
            episodes -= 1
            stale_count = 0
            robot.reset()
            return

        steps_done += 1
        if (steps_done % target_update == 0):
            target_network.load_state_dict(q_network.state_dict())

        robot.debug_info()
        print("Episode: ", episodes)
        print("Action: ", action)
        print("Reward: ", reward)
        print("Stale Count: ", stale_count)

    else:
        # Save Q-network to a file
        torch.save(q_network.state_dict(), "q_network.pth")
        sys.exit(0)
