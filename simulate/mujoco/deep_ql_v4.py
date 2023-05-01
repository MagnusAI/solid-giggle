from api_v4 import *
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
    [False, True],
    [0, 15, 30, 45, 60, 75, 90],
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


def load_network():
    global q_network, network_name
    try:
        q_network.load_state_dict(torch.load(
            network_name, map_location=device))
        print('Loaded network from disk')
    except:
        pass


def get_reward(state, next_state):
    global phase

    if (phase == 0):
        return phase_zero(state, next_state)
    elif (phase == 1):
        return phase_one(state, next_state)
    elif (phase == 2):
        return phase_two(state, next_state)
    elif (phase == 3):
        return phase_three(state, next_state)
    else:
        raise ValueError("Invalid phase")

# Attach to the ground
def phase_zero(state, next_state):
    global phase
    a_fixed, b_fixed, a_touch, b_touch, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_a_touch, next_b_touch, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

    fixed = a_fixed or b_fixed
    next_fixed = next_a_fixed or next_b_fixed
    tipping = arm_angle < next_arm_angle
    untipping = arm_angle > next_arm_angle
    rising = a_distance < next_a_distance or b_distance < next_b_distance
    falling = a_distance > next_a_distance or b_distance > next_b_distance

    modifier = .5
    if (not fixed):
        if (next_fixed):
            phase = 1
            return 10
        else:
            if (next_a_distance > 30 and next_b_distance > 30):
                return -100
            if ((falling or untipping) and not (rising or tipping)):
                return -1 * modifier
            return -1
    else:
        if (not next_fixed):
            return -100
        else:
            phase = 1
            return 0

# Find the wall
def phase_one(state, next_state):
    global phase
    a_fixed, b_fixed, a_touch, b_touch, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_a_touch, next_b_touch, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state


    angled = arm_angle == 90

    next_half_leveled = next_a_leveled or next_b_leveled
    next_fixed = next_a_fixed or next_b_fixed
    touching = a_touch or b_touch
    next_touching = next_a_touch or next_b_touch

    if (next_fixed):
        if ((next_a_touch and not next_a_fixed) or (next_b_touch and not next_b_fixed)):
            phase = 2
            return 100
        else:
            if (touching and not next_touching):
                return -100
            if (angled and (next_a_distance < 30 and next_b_distance < 30)):
                phase = 3
                return 100
        return -1
    else:
        return -100

# Lift module
def phase_two(state, next_state):
    global phase
    a_fixed, b_fixed, a_touch, b_touch, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_a_touch, next_b_touch, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

    next_fixed = next_a_fixed or next_b_fixed
    next_half_leveled = next_a_leveled or next_b_leveled

    touching = a_touch or b_touch
    next_touching = next_a_touch or next_b_touch

    angled = arm_angle == 90

    tipping = arm_angle < next_arm_angle
    rising = a_distance < next_a_distance or b_distance < next_b_distance

    if (next_fixed):
        if ((next_half_leveled and next_touching) or angled):
            phase = 3
            return 10
        if (rising or tipping):
            return 1
        return -1
    else:
        return -100

# Attach to the wall
def phase_three(state, next_state):
    global phase
    a_fixed, b_fixed, a_touch, b_touch, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_a_touch, next_b_touch, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

    fixed = a_fixed or b_fixed
    leveled = a_leveled and b_leveled
    half_leveled = a_leveled or b_leveled

    next_fixed = next_a_fixed or next_b_fixed
    next_leveled = next_a_leveled and next_b_leveled
    next_half_leveled = next_a_leveled or next_b_leveled

    tipping = arm_angle < next_arm_angle
    untipping = arm_angle > next_arm_angle
    angled = arm_angle == 90
    next_angled = next_arm_angle == 90
    rising = a_distance < next_a_distance or b_distance < next_b_distance
    falling = a_distance > next_a_distance or b_distance > next_b_distance

    touching = a_touch or b_touch
    next_touching = next_a_touch or next_b_touch

    leveled_fixed = (a_leveled and a_fixed) or (b_leveled and b_fixed)

    if (next_fixed and half_leveled):
        if ((next_a_fixed and next_a_leveled) or (next_b_fixed and next_b_leveled)):
            return 100
        if (falling or untipping):
            return 1
        return -1
    else:
        return -100


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
    return next_state


# Define the hyperparameters
learning_rate = 0.1
gamma = 0.9  # Discount factor
epsilon = 0.9  # Exploration rate (epsilon-greedy)
epsilon_decay = .999
target_update = 10
steps_done = 0
episodes = 1000

done = False
robot = None
actions = []
action_idx = None
stale_count = 0
stale_limit = 5000
state = (False, False, False, False, 0, 0, 0, False, False)  # initial state
sensor_delay = 0  # Wait for sensors data to be updated

score = 0
highscore = float('-inf')
rewards = []
network_name = "q_network_v4_m1.pth"
time_limit = 10  # seconds
episode_start_time = 0

state_history = []
state_history_limit = 10
success = 0
phase = 0


def controller(model, data):
    global robot, actions, action_idx, stale_count, stale_limit, state, sensor_delay, score, done, steps_done, episodes, epsilon, epsilon_decay, target_update, target_network, q_network, optimizer, loss_fn, highscore, rewards, network_name, time_limit, episode_start_time, state_history, state_history_limit, success, phase

    if (robot is None):
        robot = LappaApi(data)
        load_network()
        return
    else:
        robot.update_data(data)
        if (phase == 0):
            epsilon = 0.1
        else:
            epsilon = 0.75

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
            stale_count += 1
            sensor_delay = 1

            robot.lock()
            if (next_state != state):
                robot.unlock()
                stale_count = 0

            if (not robot.is_locked() or stale_count == stale_limit):
                next_state_tensor = torch.tensor(
                    [next_state], dtype=torch.float32, device=device)

                reward = get_reward(state, next_state)

                # Penalize revisiting states
                state_history.append((state, reward))
                if len(state_history) > state_history_limit:
                    state_history.pop()

                revisit_count = sum(
                    1 for s, r in state_history if s == next_state)

                if ((stale_count == stale_limit) or (data.time - episode_start_time > time_limit) or (revisit_count > 1)):
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

                rewards.append(reward)
                actions.append(action)
                score += reward

                if (reward == 100 or reward == -100):
                    if (reward == 100):
                        success += 1
                    if (score >= highscore):
                        print_info(robot, episodes, epsilon, stale_count,
                                   score, highscore, actions, rewards)
                        print("Episodes:", episodes)
                        print("Phase:", phase)
                        print("Success:", success)
                        time = round(data.time / 60, 2)
                        print("Time:", time, "minutes\n")

                    highscore = max(highscore, score)
                    episodes -= 1
                    print("Episodes: ", episodes, "Time: ",
                          round(data.time/60, 2), end="\r")
                    episode_start_time = data.time
                    actions = []
                    rewards = []
                    stale_count = 0
                    score = 0
                    phase = 0
                    robot.reset()
                    state = (False, False, False, False, 0, 0, 0, False, False)
                    return

                steps_done += 1
                if (steps_done % target_update == 0):
                    target_network.load_state_dict(q_network.state_dict())
        else:
            print("Total time:", round(data.time /
                  60, 2), "highscore:", highscore)
            print("Success:", success)
            # Save the network
            torch.save(q_network.state_dict(), network_name)
            sys.exit(0)
    else:
        sensor_delay -= 1


def print_info(robot, episodes, epsilon, stale_count, score, highscore, actions, rewards):
    robot.debug_info()
    print("Episode", episodes)
    print("Epsilon:", epsilon)
    print("Stale count:", stale_count)
    print("Score:", score, "Highscore:", highscore)
    print("Actions:", actions)
    print("Rewards:", rewards)
