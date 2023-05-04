from api_v35 import *
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


# def get_reward(state, next_state):
#     a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
#     next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

#     fixed = a_fixed and b_fixed
#     next_fixed = next_a_fixed and next_b_fixed
#     leveled = a_leveled and b_leveled
#     next_leveled = next_a_leveled and next_b_leveled
#     half_leveled = a_leveled or b_leveled
#     next_half_leveled = next_a_leveled or next_b_leveled
#     rising = a_distance < next_a_distance or b_distance < next_b_distance or arm_angle < next_arm_angle
#     falling = a_distance > next_a_distance or b_distance > next_b_distance or arm_angle > next_arm_angle
#     angled = arm_angle == 90
#     next_angled = next_arm_angle == 90

#     if (next_a_distance > 30 and next_b_distance > 30):
#         return -100

#     modifier = .25
#     reward = -1

#     if ((next_a_fixed and next_a_leveled) or (next_b_fixed and next_b_leveled)):
#         reward += 6

#     if ((a_fixed and a_leveled and not next_a_fixed) or (b_fixed and b_leveled and not next_b_fixed)):
#         return -100

#     if ((a_fixed and b_fixed) and (not next_a_fixed or not next_b_fixed)):
#         reward += 2

#     if (not fixed):
#         if (next_fixed):
#             reward += 6
#         else:
#             if (falling):
#                 if (not rising):
#                     reward += 2
#     else:
#         if (not next_fixed):
#             return -100
#         else:
#             if (half_leveled):
#                 if (next_leveled):
#                     if (next_a_fixed and next_b_fixed):
#                         return 100
#                     else:
#                         if (falling):
#                             reward *= modifier
#                 else:
#                     if (not next_half_leveled):
#                         reward -= 4
#             else:
#                 if (next_half_leveled):
#                     reward += 2
#                 else:
#                     if (rising):
#                         reward *= modifier

#     return reward

def get_reward(state, next_state):
    global stage
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

    goal_condition = next_a_fixed and next_b_fixed and next_a_leveled and next_b_leveled

    if (goal_condition):
        return 100

    reward = 0

    # Punish for unfixating a module that is leveled (Great punish) or if the other module is not fixated (great punish)
    if (a_fixed and not next_a_fixed and (not b_fixed or b_leveled)) or (b_fixed and not next_b_fixed and (not a_fixed or a_leveled)):
        return -100

    if (next_a_distance > 30 and next_b_distance > 30):
        return -100

    # Reward for fixating a module (medium reward)
    if (not a_fixed and next_a_fixed) or (not b_fixed and next_b_fixed):
        if (stage == 0):
            return 100
        reward += 1

    # Reward for fixating a module that is leveled (Great reward)
    if (not a_leveled and next_a_leveled) or (not b_leveled and next_b_leveled):
        if (stage == 2):
            return 100
        reward += 2

    # Reward for unfixating a module if the other module is fixated and leveled (small)
    if (a_fixed and b_leveled and not next_a_fixed) or (b_fixed and a_leveled and not next_b_fixed):
        reward += 0.5

    # Punish for fixating a module that isn’t leveled if the other module if fixated and leveled (small punish)
    if (a_fixed and b_leveled and not a_leveled) or (b_fixed and a_leveled and not b_leveled):
        reward -= 0.5

    # Reward for leveling a module (small reward)
    if (not a_leveled and next_a_leveled) or (not b_leveled and next_b_leveled):
        if (stage == 1):
            return 100
        if (stage == 4 and next_a_leveled and next_b_leveled):
            return 100
        reward += 0.5

    # Reward for unfixating an unlevelled module while the other module is leveled and fixated (medium reward)
    if (a_fixed and not a_leveled and b_leveled and not next_a_fixed) or (b_fixed and not b_leveled and a_leveled and not next_b_fixed):
        if (stage == 3):
            return 100
        reward += 1

    # Reward for rising when neither module is leveled if one of the module is fixated (small reward)
    if not (a_leveled or b_leveled) and (a_fixed or b_fixed) and (next_arm_angle > arm_angle):
        reward += 0.5

    # Reward for falling if neither module if fixated (small reward)
    if not (a_fixed or b_fixed) and (next_arm_angle < arm_angle):
        reward += 0.5

    # Punish if a module is not leveled after it previously was (medium punish)
    if (a_leveled and not next_a_leveled) or (b_leveled and not next_b_leveled):
        reward -= 1

    # Reward if a module is lowering distance while being leveled (small reward)
    if (a_leveled and a_distance > next_a_distance) or (b_leveled and b_distance > next_b_distance):
        reward += 0.5

    # Punish for not falling if neither module is fixed (small punish)
    if not (a_fixed or b_fixed) and (next_arm_angle >= arm_angle):
        reward -= 0.5

    return reward


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
    return next_state


# Define the hyperparameters
learning_rate = 0.25
gamma = 0.9  # Discount factor
epsilon = 0.95  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.9999
target_update = 10
steps_done = 0
episodes = 10000

done = False
robot = None
actions = []
action_idx = None
stale_count = 0
stale_limit = 1000
state = (False, False, 0, 0, 0, False, False)  # initial state
sensor_delay = 0  # Wait for sensors data to be updated

score = 0
highscore = float('-inf')
rewards = []
network_name = "q_network_simple.pth"
episode_time_limit = 15  # seconds
episode_start_time = 0

state_history = []
state_history_limit = 10
success = 0
stage = 0

training_time_limit = 60 * 2  # seconds


def controller(model, data):
    global robot, actions, action_idx, stale_count, stale_limit, state, sensor_delay, score, done, steps_done, episodes, epsilon, epsilon_decay, target_update, target_network, q_network, optimizer, loss_fn, highscore, rewards, network_name, episode_time_limit, episode_start_time, state_history, state_history_limit, success, stage

    if (robot is None):
        robot = LappaApi(data)
        load_network()
        return
    else:
        robot.update_data(data)

    if ((episodes <= 0) or (training_time_limit > 0 and data.time > training_time_limit)):
        done = True

    if (sensor_delay == 0):
        if (not done):
            state_tensor = torch.tensor(
                [state], dtype=torch.float32, device=device)

            if (not robot.is_locked()):
                if (np.random.rand() < epsilon):
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

                if ((stale_count == stale_limit) or (data.time - episode_start_time > episode_time_limit) or (revisit_count > 1)):
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
                    if (success % 100 == 0):
                        stage += 1
                    if (score >= highscore):
                        print_info(robot, episodes, epsilon, stale_count,
                                   score, highscore, actions, rewards)
                        print("Episodes:", episodes)
                        print("Success:", success)
                        time = round(data.time / 60, 2)
                        print("Time:", time, "minutes\n")

                    highscore = max(highscore, score)
                    episodes -= 1
                    print("Episodes: ", episodes, "Time: ",
                          round(data.time/60, 2), "minutes", end="\r")
                    episode_start_time = data.time
                    actions = []
                    rewards = []
                    stale_count = 0
                    score = 0
                    stage = 0
                    robot.reset()
                    state = (False, False, 0, 0, 0, False, False)
                    return

                steps_done += 1
                if (steps_done % target_update == 0):
                    target_network.load_state_dict(q_network.state_dict())
        else:
            print("")
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
