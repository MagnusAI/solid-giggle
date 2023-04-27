from api_v3 import *
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
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

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

    leveled_fixed = (a_leveled and a_fixed) or (b_leveled and b_fixed)

    # Fixating a module
    if ((not a_fixed and next_a_fixed) or (not b_fixed and next_b_fixed)):
        # print("fixating a module")
        if (not fixed):
            return 1
        else:
            a_module = (not a_fixed and next_a_fixed)
            b_module = (not b_fixed and next_b_fixed)
            if ((a_module and next_a_leveled) or (b_module and next_b_leveled)):
                return 10
            else:
                return -10

    # Unfixating a module
    if ((a_fixed and not next_a_fixed) or (b_fixed and not next_b_fixed)):
        # print("unfixating a module")
        if (not next_fixed):
            return -100
        else:
            a_module = (a_fixed and not next_a_fixed)
            b_module = (b_fixed and not next_b_fixed)
            if ((a_module and not a_leveled) or (b_module and not b_leveled)):
                return 1
            else:
                return -100

    # Lifting a module
    if (rising or tipping):
        # print("lifting a module")
        if (not fixed):
            if (a_distance == 999 and b_distance == 999):
                return -100
            else:
                return -1
        else:
            if (not leveled_fixed):
                if (not angled):
                    return 1
                else:
                    if (rising):
                        return -1
            else:
                if (leveled):
                    return -10
                else:
                    return 1

    # Lowering a module
    if (falling or untipping):
        #print("lowering a module")
        if (not fixed):
            return 1
        else:
            if (not leveled_fixed):               
                if (half_leveled and not next_half_leveled):
                    return -10
                if (angled and half_leveled and (next_a_distance < 16 and next_b_distance < 16)):
                    return 1
            else:
                if (leveled):
                    return 1
                else:
                    return -10

    # Leveling a module
    if ((not leveled and next_leveled) or (not half_leveled and next_half_leveled)):
        # print("leveling a module")
        return 10
    elif ((leveled and not next_leveled) or (half_leveled and not next_half_leveled)):
        # print("unleveling a module")
        return -100

    return -1


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
    return next_state


# Define the hyperparameters
learning_rate = 0.1
gamma = 0.9
epsilon = 0.9  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.999
target_update = 10
steps_done = 0
episodes = 1000

done = False
robot = None
actions = []
action_idx = None
stale_count = 0
stale_limit = 5000
state = (False, False, 0, 0, 0, False, False)  # initial state
sensor_delay = 0  # Wait for sensors data to be updated

score = 0
highscore = float('-inf')
rewards = []
network_name = "q_network_v3.pth"
time_limit = 30  # seconds
episode_start_time = 0

state_history = []
state_history_limit = 10

# todo = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6,
#         6, 6, 6, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4]

# todo.reverse()


def controller(model, data):
    global robot, actions, action_idx, stale_count, stale_limit, state, sensor_delay, score, done, steps_done, episodes, epsilon, epsilon_decay, target_update, target_network, q_network, optimizer, loss_fn, highscore, rewards, network_name, time_limit, episode_start_time, state_history, state_history_limit

    if (robot is None):
        robot = LappaApi(data)
        load_network()
        return
    else:
        robot.update_data(data)

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
                    if (score > highscore):
                        print_info(robot, episodes, epsilon, stale_count,
                                   score, highscore, actions, rewards)
                        print("Time:", round(data.time, 2)/60, "minutes")
                    highscore = max(highscore, score)
                    print("Episode: ", episodes)
                    episodes -= 1
                    episode_start_time = data.time
                    actions = []
                    rewards = []
                    stale_count = 0
                    score = 0
                    robot.reset()
                    return

                steps_done += 1
                if (steps_done % target_update == 0):
                    target_network.load_state_dict(q_network.state_dict())
        else:
            print("Total time:", round(data.time, 2) /
                  60, "highscore:", highscore)
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
    # print onle last 10 actions
    print("Actions:", actions[-10:])
    print("Rewards:", rewards)
