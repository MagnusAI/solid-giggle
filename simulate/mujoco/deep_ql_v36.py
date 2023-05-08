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
    next_fixed = next_a_fixed or next_b_fixed
    level_fixed = (a_fixed and a_leveled) or (b_fixed and b_leveled)
    next_level_fixed = (next_a_fixed and next_a_leveled) or (next_b_fixed and next_b_leveled)
    double_fixed = a_fixed and b_fixed
    next_double_fixed = next_a_fixed and next_b_fixed
    next_double_level_fixed = next_a_fixed and next_b_fixed and next_a_leveled and next_b_leveled
    releasing = (a_fixed and not next_a_fixed) or (b_fixed and not next_b_fixed)
    levelling = (not a_leveled and next_a_leveled) or (not b_leveled and next_b_leveled)
    unlevelling = (a_leveled and not next_a_leveled) or (b_leveled and not next_b_leveled)
    a_rising = a_distance < next_a_distance
    b_rising = b_distance < next_b_distance
    a_falling = a_distance > next_a_distance
    b_falling = b_distance > next_b_distance
    tipping = arm_angle < next_arm_angle
    untipping = arm_angle > next_arm_angle 

    rewards = -1

    if (next_double_level_fixed):
        return 100

    if (not fixed and next_fixed):
        rewards += 10
    
    if (not fixed and ((b_falling or a_falling))):
        rewards += 1.5
    
    if (not fixed and ((b_rising or a_rising))):
        rewards -= 1.5

    if (not fixed and next_a_distance > 30 and next_b_distance > 30):
        return -100
    
    if (fixed and not next_fixed):
        rewards -= 10
    
    if (not level_fixed and next_level_fixed):
        rewards += 10
    
    if (level_fixed and not next_level_fixed):
        return -100
    
    if (double_fixed and level_fixed and releasing):
        rewards += 5

    if (level_fixed and next_double_fixed and not next_double_level_fixed):
        rewards -= 10

    if (fixed and levelling):
        rewards += 5

    if (unlevelling):
        rewards -= 10
    
    if (fixed and ((not a_leveled and (a_rising or tipping)) or (not b_leveled and (b_rising or tipping)))):
        rewards += 2
    
    if (fixed and ((a_leveled and (a_falling or untipping)) or (b_leveled and (b_falling or untipping)))):
        rewards += 2

    return rewards


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
    return next_state


# Define the hyperparameters
learning_rate = 0.1
gamma = 0.9  # Discount factor
epsilon = 0.9  # Exploration rate (epsilon-greedy)
epsilon_decay = 0.9999
target_update = 5
steps_done = 0
episodes = 10000

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
network_name = "q_network_v36_new.pth"
episode_time_limit = 15  # seconds
episode_start_time = 0

state_history = []
state_history_limit = 10
success = 0
stage = 0
learning_stage = True

training_time_limit = 60 * 60  # seconds

def set_stage(new_stage):
    global stage
    stage = new_stage

def is_learning_stage(stage):
    return True
    global success
    return success < (stage + 1) * 100 if stage >= 0 else False


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

    if ((data.time - episode_start_time > episode_time_limit) and (not done)):
        state_tensor = torch.tensor(
            [state], dtype=torch.float32, device=device)
        action = action_space[action_idx]
        next_state = robot.read_state_from_sensors()
        next_state_tensor = torch.tensor(
                   [next_state], dtype=torch.float32, device=device)
        reward_tensor = torch.tensor(
                   [-100], dtype=torch.float32, device=device).squeeze()
        current_q_value = q_network(state_tensor)[0][action_idx]
        with torch.no_grad():
            next_q_value = torch.max(target_network(next_state_tensor))
        expected_q_value = reward_tensor + gamma * next_q_value
        loss = loss_fn(current_q_value, expected_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        rewards.append(-100)
        actions.append(action)
        score -= 100
        if (score >= highscore):
            print_info(robot, episodes, epsilon, stale_count,
                        score, highscore, actions, rewards)
            print("Episodes:", episodes)
            print("Success:", success)
            time = round(data.time / 60, 2)
            print("Time:", time, "minutes\n")
            print("Stage:", stage)
        highscore = max(highscore, score)
        episodes -= 1
        episode_start_time = round(data.time, 1)
        actions = []
        rewards = []
        stale_count = 0
        score = 0
        robot.reset()
        state = (False, False, 0, 0, 0, False, False)
        return


    if (action_idx is not None):
        print("TST: ", round(data.time, 2), round(episode_start_time, 2), round(data.time - episode_start_time, 2), "Action: ", action_space[action_idx], end="\r")
    #print("Episodes: ", episodes, "Time: ", round(data.time/60, 2), "minutes, epsilon", round(epsilon, 3), "success:", success,  end="\r")

    if (sensor_delay == 0):
        if (not done):
            state_tensor = torch.tensor(
                [state], dtype=torch.float32, device=device)

            if (not robot.is_locked() and is_learning_stage(stage)):
                if (np.random.rand() < epsilon):
                    action_idx = np.random.choice(action_dimensions)
                else:
                    with torch.no_grad():
                        action_idx = torch.argmax(
                            q_network(state_tensor)).item()

            action = action_space[action_idx]
            next_state = perform_action(robot, action)
            stale_count += 1
            sensor_delay = 0
            is_stopping_rotation = action == 'stop_a_rotation' or action == 'stop_b_rotation'

            robot.lock()
            if (next_state != state or is_stopping_rotation):
                robot.unlock()
                stale_count = 0

            if (not robot.is_locked() or stale_count == stale_limit):
                next_state_tensor = torch.tensor(
                    [next_state], dtype=torch.float32, device=device)

                reward = get_reward(state, next_state)

                if (is_stopping_rotation):
                    reward = 0

                # Penalize revisiting states
                state_history.append((state, reward))
                if len(state_history) > state_history_limit:
                    state_history.pop()

                revisit_count = sum(
                    1 for s, r in state_history if s == next_state)
                
                if ((stale_count == stale_limit)):
                    reward = -10

                if ((revisit_count > 1)):
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

                if (is_learning_stage(stage)):
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
                            #epsilon = 0.6
                    if (score >= highscore):
                        print_info(robot, episodes, epsilon, stale_count,
                                   score, highscore, actions, rewards)
                        print("Episodes:", episodes)
                        print("Success:", success)
                        time = round(data.time / 60, 2)
                        print("Time:", time, "minutes\n")
                        print("Stage:", stage)

                    highscore = max(highscore, score)
                    episodes -= 1
                    episode_start_time = round(data.time, 1)
                    actions = []
                    rewards = []
                    stale_count = 0
                    score = 0
                    robot.reset()
                    state = (False, False, 0, 0, 0, False, False)
                    return

                steps_done += 1
                if (steps_done % target_update == 0):
                    target_network.load_state_dict(q_network.state_dict())
            else:
                if ((data.time - episode_start_time > episode_time_limit)):
                    stale_count = stale_limit
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
