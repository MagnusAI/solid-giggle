import sys
import numpy as np
import itertools
import os
from api_v2 import *

# Define state space
state_space = list(itertools.product([False, True], repeat=6))

# Define action space
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward']

# State and action dimensions
state_dim = len(state_space)
action_dim = len(action_space)

# Load Q-table from the file if it exists, otherwise initialize a new one
q_table_file = "q_table.npy"

if os.path.exists(q_table_file):
    Q = np.load(q_table_file)
else:
    Q = np.zeros((state_dim, action_dim))

# Function to get the index of a state


def get_state_index(state):
    return state_space.index(tuple(state))

# Function to perform an action on the robot and read the next state from the sensors


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.get_state()
    return next_state

# Function to get the reward given current state, action, and next state


def get_reward(state, action, next_state):
    init_condition = not any(next_state)
    goal_conditions = (next_state[0] and next_state[1]
                       and next_state[4] and next_state[5])
    reset_conditions = (
        not next_state[0] and not next_state[1]) and not init_condition

    if goal_conditions:
        return 500
    elif reset_conditions and state != (False, False, False, False, False, False):
        return -500
    else:
        height_reward = 0
        if next_state[4] and not state[4]:
            height_reward += 1
        elif not next_state[4] and state[4]:
            height_reward -= 2
        if next_state[5] and not state[5]:
            height_reward += 1
        elif not next_state[5] and state[5]:
            height_reward -= 2

        return height_reward if height_reward != 0 else -1


# Q-learning algorithm parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000

# Initialize the robot
robot = None

# Q-learning algorithm

state = (False, False, False, False, False, False)


def controller(model, data):
    global robot, state, epsilon, Q
    if (robot is None):
        robot = LappaApi(data)
        return
    else:
        robot.update_data(data)

    done = False

    if (data.time > (60 * 10)):
        done = True

    if (not done and not robot.locked):
        state_idx = get_state_index(state)

        if np.random.rand() < epsilon:
            action_idx = np.random.choice(action_dim)
        else:
            action_idx = np.argmax(Q[state_idx])

        action = action_space[action_idx]
        next_state = perform_action(robot, action)

        if (next_state != state):
            robot.unlock()
            print("Action: " + action, " State: " + str(state))
            if (state[4] or state[5]):
                print("#####################################################################################")

        next_state_idx = get_state_index(next_state)

        reward = get_reward(state, action, next_state)

        # Update Q-table
        Q[state_idx, action_idx] = Q[state_idx, action_idx] + alpha * \
            (reward + gamma *
             np.max(Q[next_state_idx]) - Q[state_idx, action_idx])

        # Add epsilon_decay as a parameter
        epsilon_decay = 0.999

        # Update epsilon in the controller function after performing an action
        epsilon = max(epsilon * epsilon_decay, 0.01)

        state = next_state

        if reward == 500 or reward == -500:
            robot.reset()
    else:
        # Round all values in Q-table to 2 decimal places
        Q = np.round(Q, 2)
        print(Q)
        # Save Q-table to a file
        np.save("q_table.npy", Q)
        sys.exit(0)
