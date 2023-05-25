from api_v36 import *
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
    [0, 15, 30, 45, 60, 75, 90],
    [999, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
    [999, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
    [False, True],
    [False, True]
))

neutral_actions = ['stop_a_rotation', 'stop_b_rotation']

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

# Define the hyperparameters
learning_rate = 0.001
gamma = 0.95  # Discount factor
epsilon = 0.75  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.999
target_update = 10
steps_done = 0
episodes = 100000

done = False
robot = None
actions = []
action_idx = None
stale_count = 0
stale_limit = 5000
state = (False, False, 0, 0, 0, False, False)  # initial state

score = 0
highscore = float('-inf')
rewards = []
network_name = "q_network_v36_hyper_v1.pth"
episode_time_limit = 15  # seconds
episode_start_time = 0

state_history = []
state_history_limit = 10
success = 0

training_time_limit = 60 * 60 * 2  # seconds

def initialize(data):
    global robot
    robot = LappaApi(data)
    load_network()

def update(data):
    global robot
    if (robot is None):
        initialize(data)
        return
    else:
        robot.update_data(data)

def get_stale_count():
    global stale_count
    return stale_count


def controller(model, data):
    global robot, done, episodes, training_time_limit,epsilon, success, action_space, action_idx, state, stale_count, stale_limit, q_network, network_name

    update(data)
    done = (episodes <= 0) or (training_time_limit > 0 and data.time > training_time_limit)
    episode_timedout = data.time - episode_start_time > episode_time_limit

    print("Episodes: ", episodes, "Time: ", round(data.time/60, 2), "minutes, epsilon", round(epsilon, 3), "success:", success,  end="\r")

    if (not done):
        if (episode_timedout):
            action = action_space[action_idx]
            next_state = robot.get_state()
            update_q_network(state, action_idx, -100, next_state)
            update_statistics(action, -100, next_state)
            handle_reward(-100)
        else:
            action = choose_action()
            next_state = perform_action(robot, state, action)

            if (not robot.is_locked() or stale_count == stale_limit):
                reward = calculate_reward(state, next_state, action)
                update_q_network(state, action_idx, reward/10, next_state)
                update_statistics(action, reward, next_state)
                handle_reward(reward)
    else:
        print("-------------------- DONE --------------------")
        print_statistics()
        # Save the network
        torch.save(q_network.state_dict(), network_name)
        sys.exit(0)

def load_network():
    global q_network, network_name
    try:
        q_network.load_state_dict(torch.load(
            network_name, map_location=device))
        print('Loaded network from disk')
    except:
        pass

def get_reward(state, next_state):
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state
    next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_levelled, next_b_levelled = next_state

    fixed, next_fixed = a_fixed or b_fixed, next_a_fixed or next_b_fixed
    level_fixed, next_level_fixed = (a_fixed and a_levelled) or (b_fixed and b_levelled), (next_a_fixed and next_a_levelled) or (next_b_fixed and next_b_levelled)
    double_fixed, next_double_fixed, next_double_level_fixed = a_fixed and b_fixed, next_a_fixed and next_b_fixed, next_a_fixed and next_b_fixed and next_a_levelled and next_b_levelled
    releasing, levelling, unlevelling = (a_fixed and not next_a_fixed) or (b_fixed and not next_b_fixed), (not a_levelled and next_a_levelled) or (not b_levelled and next_b_levelled), (a_levelled and not next_a_levelled) or (b_levelled and not next_b_levelled)
    a_rising, b_rising, a_falling, b_falling = a_distance < next_a_distance, b_distance < next_b_distance, a_distance > next_a_distance, b_distance > next_b_distance
    tipping, untipping = arm_angle < next_arm_angle, arm_angle > next_arm_angle

    reward = -1
    
    if next_double_level_fixed: return 100
    if not fixed and next_fixed: reward += 10
    if not fixed and (b_falling or a_falling): reward += 1.5
    if not fixed and (b_rising or a_rising): reward -= 1.5
    if not fixed and next_a_distance > 30 and next_b_distance > 30: return -100
    if fixed and not next_fixed: return -100
    if not level_fixed and next_level_fixed: reward += 10
    if level_fixed and not next_level_fixed: return -100
    if double_fixed and next_level_fixed and releasing: reward += 10
    if level_fixed and next_double_fixed and not next_double_level_fixed: reward -= 10
    if fixed and levelling: reward += 2
    if unlevelling: reward -= 5
    if fixed and ((not a_levelled and (a_rising or tipping)) or (not b_levelled and (b_rising or tipping))): reward += 2
    if fixed and ((a_levelled and (a_falling or untipping)) or (b_levelled and (b_falling or untipping))): reward += 2

    return reward

def perform_action(robot, state, action):
    global stale_count
    robot.lock()
    robot.perform_action(action)
    stale_count += 1
    next_state = robot.get_state()
    if (next_state != state or (action in neutral_actions)):
        robot.unlock()
        stale_count = 0
    return next_state

def calculate_reward(state, next_state, action):
    global stale_count, stale_limit, neutral_actions

    reward = get_reward(state, next_state)

    if (action in neutral_actions):
        return -0.5

    # if (is_revisit_looping(state, next_state, reward)):
    #     return -100

    if ((stale_count == stale_limit)):
        return -50
    
    return reward

def handle_reward(reward):
    global robot, success, steps_done, target_update,target_network, q_network, state_history, state_history_limit
    if (reward == 100 or reward == -100):
        end_episode()
    else:
        steps_done += 1
        if (steps_done % target_update == 0):
            target_network.load_state_dict(q_network.state_dict())

def is_revisit_looping(state, next_state, reward):
    global state_history, state_history_limit
    state_history.append((state, reward))
    if len(state_history) > state_history_limit:
        state_history.pop()
    revisit_count = sum(
        1 for s, r in state_history if s == next_state)
    return (revisit_count > 1)

def choose_action():
    global action_idx, action_space
    action_idx = get_action_idx()
    return action_space[action_idx]

def get_action_idx():
    global robot, state, action_idx, device, epsilon, action_dimensions, q_network
    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)

    if (not robot.is_locked()):
        if (np.random.rand() < epsilon):
            return np.random.choice(action_dimensions)
        else:
            with torch.no_grad():
                return torch.argmax(q_network(state_tensor)).item()
    else:
        return action_idx

def update_q_network(state, action_idx, reward, next_state):
    global gamma, epsilon_decay, epsilon, rewards, actions, score, success, device, q_network, target_network, optimizer, loss_fn
    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
    next_state_tensor = torch.tensor([next_state], dtype=torch.float32, device=device)
    reward_tensor = torch.tensor([reward], dtype=torch.float32, device=device).squeeze()

    current_q_value = q_network(state_tensor)[0][action_idx]
    with torch.no_grad():
        next_q_value = torch.max(target_network(next_state_tensor))
    expected_q_value = reward_tensor + gamma * next_q_value

    loss = loss_fn(current_q_value, expected_q_value)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def update_statistics(action, reward, next_state):
    global epsilon, state, score, actions, rewards, success
    epsilon *= epsilon_decay
    state = next_state
    rewards.append(reward)
    actions.append(action)
    score += reward
    update_highscore(score)
    if (reward == 100):
        success += 1
    

def update_highscore(score):
    global highscore
    if (score >= highscore):    
        highscore = max(highscore, score)
        print_statistics()
    else:
        highscore = max(highscore, score)

def print_statistics():
    global highscore, state, success, episodes, epsilon, stale_count, score, actions, rewards, robot
    print("-------------------- STATISTICS --------------------")
    print("Actions:", actions)
    print("Rewards:", rewards)
    print("Stale count:", stale_count)
    print("Epsilon:", epsilon)
    print("Success:", success)
    time = round(robot.get_data().time / 60, 2)
    print("Time:", time, "minutes")
    print("Episodes:", episodes)
    print("State:", state)
    print("Highscore:", highscore)

def end_episode():
    global episodes, episode_start_time, actions, rewards, stale_count, score, robot, state
    episodes -= 1
    episode_start_time = round(robot.get_data().time, 1)
    actions = []
    rewards = []
    stale_count = 0
    score = 0
    robot.reset()
    state = (False, False, 0, 0, 0, False, False)