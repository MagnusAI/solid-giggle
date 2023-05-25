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
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward']

# Define the state space
state_space = list(itertools.product(
    [False, True],
    [False, True],
    [False, True],
    [999, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
    [999, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
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
    next_state = robot.get_state()
    return next_state


def should_reset(state, next_state):
    if not next_state[0] and not next_state[1] and (state[0] or state[1]):
        return True
    if (state[3] > 900 and state[4] > 900):
        return True
    return False


def get_reward(state, next_state):
    a_fixed, b_fixed, lifted, a_distance, b_distance, a_levelled, b_levelled = state
    next_a_fixed, next_b_fixed, next_lifted, next_a_distance, next_b_distance, next_a_levelled, next_b_levelled = next_state

    fixed = a_fixed or b_fixed
    levelled = a_levelled and b_levelled
    half_levelled = a_levelled or b_levelled

    next_fixed = next_a_fixed or next_b_fixed
    next_levelled = next_a_levelled and next_b_levelled
    next_half_levelled = next_a_levelled or next_b_levelled

    rising = a_distance < next_a_distance or b_distance < next_b_distance
    falling = a_distance > next_a_distance or b_distance > next_b_distance

    if (next_a_fixed and next_b_fixed and levelled):
        return 100

    if (not fixed and falling and not rising):
        return 1
    elif (not fixed and not next_fixed):
        return -100
    elif (fixed and rising and not half_levelled):
        return 1
    elif (fixed and next_half_levelled and falling):
        return 1
    elif (fixed and next_half_levelled and rising):
        return -1
    elif (next_half_levelled and ((next_a_fixed and not a_fixed) or (next_b_fixed and not b_fixed))):
        return 1
    elif (half_levelled and (a_fixed and b_fixed) and ((a_levelled and not next_b_fixed) or (b_levelled and not next_a_fixed))):
        return 1
    elif (half_levelled and fixed and next_levelled):
        return 0
    elif (levelled and not next_levelled):
        return -1
    elif (levelled and falling):
        return 1
    else:
        if (not fixed and next_fixed):
            return 1
        elif (fixed and not next_fixed):
            return -100
        else:
            if (fixed and next_half_levelled and not half_levelled):
                return 1
            elif (fixed and not next_half_levelled and half_levelled and not next_levelled):
                return -1

            if (fixed and next_half_levelled and falling):
                return 1
            elif (fixed and next_half_levelled and rising):
                return -1

    return -1


# Define the hyperparameters
learning_rate = 0.1
gamma = 0.9
epsilon = 0.9  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.999
target_update = 10
steps_done = 0
robot = None
state = (False, False, False, 0, 0, False, False)
episodes = 10
action_idx = None
stale_count = 0
stale_limit = 10

success_count = 0
score = 0
#high_sscore starts at lowest int value
high_score = float('-inf')
actions = []
scores = []

state_history = []
state_history_limit = 10
revisit_penalty = 20

sensor_delay = 0  # Wait for sensors to update


def controller(model, data):
    global robot, state, q_network, target_network, optimizer, epsilon, episodes, steps_done, action_idx, stale_count, stale_limit, success_count, score, high_score, actions, state_history, state_history_limit, revisit_penalty, sensor_delay, scores

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

    if (sensor_delay == 0):
        if (not done):
            state_tensor = torch.tensor(
                [state], dtype=torch.float32, device=device)

            if (not robot.is_locked()):
                if np.random.rand() < epsilon:
                    action_idx = np.random.choice(action_dimensions)
                else:
                    with torch.no_grad():
                        action_idx = torch.argmax(
                            q_network(state_tensor)).item()

            action = action_space[action_idx]
            next_state = perform_action(robot, action)
            sensor_delay = 100

            robot.lock()
            if (next_state != state):
                robot.unlock()
                stale_count = 0
            else:
                stale_count += 1

            if (not robot.is_locked() or stale_count == stale_limit):
                next_state_tensor = torch.tensor(
                    [next_state], dtype=torch.float32, device=device)

                reward = get_reward(state, next_state)

                # Penalize revisiting states
                state_history.append((state, reward))
                if len(state_history) > state_history_limit:
                    state_history.pop(0)

                revisit_count = sum(
                    1 for s, r in state_history if s == next_state)
                if revisit_count > 1:
                    reward -= revisit_penalty * (revisit_count - 1)

                if (stale_count == stale_limit):
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

                score += reward
                scores.append(reward)
                actions.append(action)

                if (reward == 100 or reward == -100):
                    if (reward == 100):
                        success_count += 1
                    episodes -= 1
                    print(
                        "-----------------------------------------------------------------------------------------------------------------------")
                    print("Actions: ", actions)
                    print("State: ", state)
                    print("Episode:", episodes)
                    print("scores:", scores)
                    print("Highscore:", high_score)
                    print("Stale count: ", stale_count)
                    if (score > high_score):
                        high_score = score
                    score = 0
                    actions = []
                    scores = []
                    robot.reset()
                    return

                steps_done += 1
                if (steps_done % target_update == 0):
                    target_network.load_state_dict(q_network.state_dict())

        else:
            print("Success rate: ", success_count /
                  100, "Highscore: ", high_score)
            # Save Q-network to a file
            
    else:
        sensor_delay -= 1
