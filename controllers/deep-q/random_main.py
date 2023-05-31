import mujoco as mj
from mujoco.glfw import glfw
from api import LappaApi
import os
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
import numpy as np

# Open/create file for storing Q-value distributions
q_values_file = open('q_value_distributions_random.txt', 'w')

# Define the action space
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward', 'stop_a_rotation', 'stop_b_rotation']

neutral_actions = ['stop_a_rotation', 'stop_b_rotation']

# Define the state space
state_space = list(itertools.product(
    [False, True],
    [False, True],
    [0, 15, 30, 45, 60, 75, 90],
    [999, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
    [999, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30],
    [False, True],
    [False, True],
))

# Define state_dimensions
state_dimensions = len(state_space)
action_dimensions = len(action_space)

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

# Define the neural network
input_dimensions = len(state_space[0])
output_dimensions = len(action_space)
q_network = DQN(input_dimensions, output_dimensions).to(device)
target_network = DQN(input_dimensions, output_dimensions).to(device)
target_network.load_state_dict(q_network.state_dict())
target_network.eval()

# Define the optimizer and loss function
optimizer = optim.Adam(q_network.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Define the hyperparameters
learning_rate = 0.01
gamma = 0.95  # Discount factor
epsilon = 1  # Exploration rate (epsilon-greedy)
epsilon_decay = 1
target_update = 10
steps_done = 0

network_name = "q_network_random.pth"
current_state = None
action_idx = None
episode_score = 0
rewards = []
episode_scores = []
latest_reward = None
stale_count = 0

episode_counter = 0
max_episodes = 1000
episode_time_limit = 20

def controller(model, data):
    global robot, action_idx, episode_score, rewards, current_state, stale_count, latest_reward
    robot.update_data(data)
    robot.update_pressure()
    if (current_state == None):
        current_state = robot.get_state()
    action = get_action(robot, current_state)
    next_state = perform_action(robot, action)
    if (next_state != current_state or action in neutral_actions):
        robot.unlock()
        if (action in neutral_actions):
            reward = -0.1
        else:
            reward = get_reward(current_state, next_state)
        update_q_network(current_state, next_state, action_idx, reward)
        update_target_network(target_network, q_network)
        current_state = next_state
        episode_score += reward
        rewards.append(reward)
        latest_reward = reward
        stale_count = 0
    else:
        stale_count += 1
        if (stale_count > 5000):
            reward = -1
            update_q_network(current_state, next_state, action_idx, reward)
            update_target_network(target_network, q_network)
            episode_score += reward
            rewards.append(reward)
            latest_reward = reward
            stale_count = 0
            robot.unlock()


# Get the next action using an epsilon-greedy policy if the robot isn't locked
def get_action(robot, state):
    global epsilon, action_idx, action_space, action_dimensions, device, q_network
    state_tensor = torch.tensor([state], dtype=torch.float32, device=device)

    if (not robot.is_locked()):
        if (np.random.rand() < epsilon):
            action_idx = np.random.choice(action_dimensions)
        else:
            with torch.no_grad():
                action_idx = torch.argmax(q_network(state_tensor)).item()
    
    return action_space[action_idx]

def perform_action(robot, action):
    if (not robot.is_locked()):
        robot.lock()
        robot.perform_action(action)
    return robot.get_state()

def get_reward(state, next_state):
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state
    next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_levelled, next_b_levelled = next_state

    fixed, next_fixed = a_fixed or b_fixed, next_a_fixed or next_b_fixed
    level_fixed, next_level_fixed = (a_fixed and a_levelled) or (b_fixed and b_levelled), (next_a_fixed and next_a_levelled) or (next_b_fixed and next_b_levelled)
    double_fixed, next_double_fixed, next_double_level_fixed = a_fixed and b_fixed, next_a_fixed and next_b_fixed, next_a_fixed and next_b_fixed and next_a_levelled and next_b_levelled
    releasing, levelling, unlevelling = (a_fixed and not next_a_fixed) or (b_fixed and not next_b_fixed), (not a_levelled and next_a_levelled) or (not b_levelled and next_b_levelled), (a_levelled and not next_a_levelled) or (b_levelled and not next_b_levelled)
    a_rising, b_rising, a_falling, b_falling = a_distance < next_a_distance, b_distance < next_b_distance, a_distance > next_a_distance, b_distance > next_b_distance
    tipping, untipping = arm_angle < next_arm_angle, arm_angle > next_arm_angle

    reward = -.1

    if next_double_level_fixed: return 500
    if not fixed and (next_a_distance > 30 and next_b_distance > 30): return -100
    if not fixed and next_fixed: reward = 1
    if (not fixed or not next_fixed) and (a_rising or b_rising): reward = -.5
    if not fixed and ((a_falling and not b_rising) or (b_falling and not a_rising)): reward += .2
    if fixed and levelling: reward += .2
    if fixed and unlevelling: reward = -.5
    if fixed and arm_angle == 90 and (a_falling or b_falling): reward += .2
    if fixed and ((a_levelled and a_rising) or (b_levelled and b_rising)): reward += .1
    if fixed and ((not a_levelled and a_rising) or (not b_levelled and b_rising)): reward += .2
    if fixed and ((a_levelled and a_falling) or (b_levelled and b_falling)): reward += .1
    if next_double_fixed and next_level_fixed: reward = 1
    if double_fixed and releasing: reward = .5
    if next_double_fixed and not next_level_fixed: reward = -.5
    if not level_fixed and next_level_fixed: reward = 1
    if (fixed and not next_fixed): reward = -1
    if level_fixed and not next_level_fixed: reward = -100

    return reward
    
    
def update_target_network(target_network, q_network):
    global steps_done, target_update
    steps_done += 1
    if (steps_done % target_update == 0):
        target_network.load_state_dict(q_network.state_dict())

def update_q_network(state, next_state, action_idx, reward):
    global gamma, device, q_network, target_network, optimizer, loss_fn
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

    # Write Q-values to file
    q_values_file.write(f"{current_q_value.item()},{expected_q_value.item()}\n")


def load_network(q_network, network_name):
    try:
        q_network.load_state_dict(torch.load(
            network_name, map_location=device))
        print('Loaded', network_name, ' from disk')
    except:
        print('No network found')
        pass

def save_network(q_network, network_name):
    torch.save(q_network.state_dict(), network_name)
    print('Saved', network_name, ' to disk')

robot_path = '../../model/index_angle.xml'
robot_controller = controller

# Define the path to the XML model file.
xml_path = os.path.join(os.path.dirname(__file__), robot_path)

# Load the Mujoco model from the XML file.
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Set the control callback for the Mujoco model.
mj.set_mjcb_control(robot_controller)

glfw.init()
window = glfw.create_window(600, 450, "Lappa Deep-Q (RANDOM)", None, None)
glfw.make_context_current(window)

# Set VSync to 1, which means that the window's buffer will be swapped with the front buffer at most once per frame.
glfw.swap_interval(1)

cam = mj.MjvCamera()
opt = mj.MjvOption()

# Create a Mujoco scene object with a maximum of 10000 geometric objects.
scene = mj.MjvScene(model, maxgeom=10000)

context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

target_fps = 60.0

# Function to reset the environment (you may need to adjust this depending on your needs)
def reset_env():
    global data, episode_counter, episode_score, rewards, robot, epsilon, epsilon_decay
    episode_counter += 1
    data = mj.MjData(model)
    robot.update_data(data)
    episode_score = 0
    rewards = []
    robot.unlock()
    epsilon *= epsilon_decay

def is_terminal_state(data):
    global robot, episode_counter, episode_time_limit, latest_reward
    a_fixed, b_fixed, arm_angle, a_range, b_range, a_levelled, b_levelled = robot.get_state() 

    terminal_condition = ((not a_fixed and not b_fixed) and (a_range > 30 and b_range > 30))
    goal_condition = ((a_fixed and b_fixed) and (a_levelled and b_levelled)) or (latest_reward == 500)
    time_limit = real_time(data.time) > episode_time_limit
    return goal_condition or terminal_condition or time_limit

def real_time(data_time):
    return data_time * 0.4

robot = LappaApi(data)
load_network(q_network, network_name)

while not glfw.window_should_close(window):
    if(episode_counter >= max_episodes):
        break

    sim_start = data.time 
    print(round(real_time(data.time),2), end="\r")

    while (data.time - sim_start < 1.0/target_fps):
        mj.mj_step(model, data)
        
    if is_terminal_state(data):
        print("Episode", episode_counter, "score:", round(episode_score,1), "final state:", robot.get_state(), "latest reward:", latest_reward)
        episode_scores.append(round(episode_score, 2))
        reset_env()

    # Get and set the width and height of the window's framebuffer.
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    # Update the scene with the current simulation data.
    mj.mjv_updateScene(model, data, opt, None, cam,
                       mj.mjtCatBit.mjCAT_ALL.value, scene)

    # Render the scene to the window's framebuffer.
    mj.mjr_render(viewport, scene, context)

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
print("Episode scores:", episode_scores)
save_network(q_network, network_name)
q_values_file.close()

