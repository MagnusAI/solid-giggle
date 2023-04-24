import sys
import numpy as np
import itertools
from api_v2 import *
import torch
import torch.nn as nn
import torch.optim as optim

# Define state space
state_space = list(itertools.product([False, True], repeat=6))

# Define action space
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward', 'stop_a_rotation', 'stop_b_rotation']

# State and action dimensions
state_dim = len(state_space)
action_dim = len(action_space)


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_dim = len(state_space[0])
output_dim = action_dim

q_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(q_net.state_dict())
target_net.eval()

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()


# Function to perform an action on the robot and read the next state from the sensors
def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
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

        fix_reward = 0
        if next_state[0] and not state[0] and next_state[4]:
            fix_reward += 1
        elif not next_state[0] and state[0] and state[4]:
            fix_reward -= 2
        elif next_state[1] and not state[1] and next_state[5]:
            fix_reward += 1
        elif not next_state[1] and state[1] and state[5]:
            fix_reward -= 2

        acc_reward = height_reward + fix_reward

        return acc_reward if acc_reward != 0 else -1


# Q-learning algorithm parameters
alpha = 0.1
gamma = 0.99
epsilon = 0.1
episodes = 1000
target_update = 100
epsilon_decay = 0.999
steps_done = 0

# Initialize the robot
robot = None

# Q-learning algorithm
state = (False, False, False, False, False, False)


def controller(model, data):
    global robot, state, q_net, target_net, optimizer, epsilon, steps_done
    if (robot is None):
        robot = LappaApi(data)
        return
    else:
        robot.update_data(data)

    done = False

    if (data.time > (60 * 5)):
        done = True

    if not done and not robot.locked:
        encoded_state = [float(s) for s in state]
        state_tensor = torch.FloatTensor(encoded_state).unsqueeze(0).to(device)

        if np.random.rand() < epsilon:
            action_idx = np.random.choice(action_dim)
        else:
            with torch.no_grad():
                action_idx = torch.argmax(q_net(state_tensor)).item()

        action = action_space[action_idx]
        next_state = perform_action(robot, action)

        if (next_state != state):
            robot.unlock()
            print("Time: " + str(round(data.time, 0)), "State: " +
                  str(state) + "    Action: " + action,)

        # Inside the controller function, after getting the next_state
        next_encoded_state = [float(s) for s in next_state]
        next_state_tensor = torch.FloatTensor(
            next_encoded_state).unsqueeze(0).to(device)

        reward = get_reward(state, action, next_state)
        reward_tensor = torch.FloatTensor([reward]).to(device)

        # Update Q-network
        current_q_value = q_net(state_tensor)[0, action_idx]
        with torch.no_grad():
            next_q_value = target_net(next_state_tensor).max(1)[0]
        expected_q_value = reward_tensor + gamma * next_q_value

        loss = loss_fn(current_q_value, expected_q_value)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update epsilon in the controller function after performing an action
        epsilon = max(epsilon * epsilon_decay, 0.01)

        state = next_state

        if reward == 500 or reward == -500:
            robot.reset()

        # Update the target Q-network in the controller function
        steps_done += 1
        if steps_done % target_update == 0:
            target_net.load_state_dict(q_net.state_dict())
    else:
        # Save Q-network to a file
        torch.save(q_net.state_dict(), "q_net.pth")
        sys.exit(0)
