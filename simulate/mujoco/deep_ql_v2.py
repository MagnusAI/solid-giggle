from api_v2 import *
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
    global q_network
    try:
        q_network.load_state_dict(torch.load(
            'q_network.pth', map_location=device))
        print('Loaded network from disk')
    except:
        pass

def get_reward(state, next_state):
    a_fixed, b_fixed, a_distance, b_distance, a_leveled, b_leveled = state
    next_a_fixed, next_b_fixed, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state

    fixed = a_fixed or b_fixed
    leveled = a_leveled and b_leveled
    half_leveled = a_leveled or b_leveled

    next_fixed = next_a_fixed or next_b_fixed
    next_leveled = next_a_leveled and next_b_leveled
    next_half_leveled = next_a_leveled or next_b_leveled

    rising = a_distance < next_a_distance or b_distance < next_b_distance
    falling = a_distance > next_a_distance or b_distance > next_b_distance

    # Reward for fixing both modules on the wall
    if (next_a_fixed and next_b_fixed and leveled):
        return 100
    
    # Punishment for entering a state where both modules are unfixed
    if (fixed and not next_fixed):
        return -100
    
    if (not fixed and (next_a_distance > 30 or next_b_distance > 30)):
        return -100

    # Reward for fixating a module when floating
    if (not fixed and next_fixed):
        return 1

    # Reward for reaching half-leveled state
    if (not half_leveled and next_half_leveled):
        return 10
    
    # Punishment for leaving half-leveled
    if (half_leveled and not next_half_leveled):
        return -10

    # Reward for fixing a module on the wall
    if (fixed and next_half_leveled and ((next_a_fixed and not a_fixed) or (next_b_fixed and not b_fixed))):
        return 10
    
    # Punishment for releasing a module from the wall
    if (((not next_a_fixed and a_fixed and a_leveled) or (not next_b_fixed and b_fixed and b_leveled))):
        return -10
    
    # Reward for releasing a module from ground when th other has been fixed on the wall
    if (a_fixed and b_fixed and half_leveled and ((not a_leveled and not next_a_fixed) or (not b_leveled and not next_b_fixed))):
        return 1
    
    # Reward and punishement for entering and leaving the leveled state
    if (not leveled and next_leveled):
        return 10
    elif ( leveled and not next_leveled):
        return -10

    # Reward for moving closer to the wall with the unfixed modules while leveled
    if (fixed and leveled and falling):
        return 1
    
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
episodes = 5000

done = False
robot = None
actions = []
action_idx = None
stale_count = 0
stale_limit = 100
state = (False, False, 0, 0, False, False)  # init_state
sensor_delay = 0  # Wait for sensors data to be updated

score = 0
highscore = float('-inf')
rewards = []

# todo = [3,6, 6, 6, 6, 6, 6, 6, 6, 6, 1, 1, 1, 2]

def controller(model, data):
    global robot, actions, action_idx, stale_count, stale_limit, state, sensor_delay, score, done, steps_done, episodes, epsilon, epsilon_decay, target_update, target_network, q_network, optimizer, loss_fn, highscore, rewards

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

                rewards.append(reward)
                actions.append(action)
                score += reward

                if (reward == 100 or reward == -100):
                    print("-------------------------------------------------------------------------------------------")
                    print("Episode", episodes)
                    print("Epsilon:", epsilon)
                    print("Stale count:", stale_count)
                    print("Score:", score, "Highscore:", highscore)
                    print("State:", state)
                    print("Actions:", actions)
                    print("Rewards:", rewards)
                    robot.debug_info()

                    episodes -= 1
                    highscore = max(highscore, score)
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
            # Save the network
            filename = "q_network.pth"
            torch.save(q_network.state_dict(), filename)
            sys.exit(0)
    else:
        sensor_delay -= 1
