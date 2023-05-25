import mujoco as mj
from mujoco.glfw import glfw
from controller import controller
from api import LappaApi
import os
from dqn import DQN
import itertools
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
from api import LappaApi

#### DQN CONTROLLER ####

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
epsilon = 0.75  # Exploration rate (epsilon-greedy)
epsilon_decay = .999
target_update = 10
steps_done = 0

# Define limits
episodes = 10000
episode_time_limit = 25  # seconds
stale_state_limit = 3000

# Global variables
network_name = "q_network_main.pth"
robot = None
stale_state_counter = 0
episode_start_time = 0
episode_counter = 0
episode_score = 0
episode_rewards = []
episode_actions = []
highscore = float('-inf')
highscore_actions = []
current_state = None
initial_state = None
action_idx = None
episodes_success = 0

def controller(model, data):
    global robot, episode_time_limit
    if (robot is None):
        initialize(model, data)
        return
    else:
        robot.update_data(data)

    simulation_time = round(data.time, 2)

    episode_timeout = simulation_time - episode_start_time > episode_time_limit and episode_time_limit > 0
    if (episode_timeout):
        end_episode(robot, data)
    else:
        execute_step(robot, data)

def initialize(model, data):
    global robot, current_state, initial_state
    robot = LappaApi(data)
    initial_state = robot.get_state()
    current_state = initial_state
    load_network()

def load_network():
    global q_network, network_name
    try:
        q_network.load_state_dict(torch.load(
            network_name, map_location=device))
        print('Loaded', network_name, ' from disk')
    except:
        print('No network found')
        pass

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

# Execute an action on the robot and return the next state
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

    if not fixed and next_fixed: reward = 1
    if (not fixed or next_a_fixed) and (a_rising or b_rising): reward = -.5
    if not fixed and ((a_falling and not b_rising) or (b_falling and not a_rising)): reward += .2
    if fixed and levelling: reward += .2
    if fixed and unlevelling: reward = -.5
    if fixed and ((a_levelled and a_rising) or (b_levelled and b_rising)): reward += .1
    if fixed and ((not a_levelled and a_rising) or (not b_levelled and b_rising)): reward += .2
    if fixed and ((a_levelled and a_falling) or (b_levelled and b_falling)): reward += .1
    if next_double_fixed and next_level_fixed: reward = .5
    if double_fixed and releasing: reward = .5
    if next_double_fixed and not next_level_fixed: reward = -.5
    if not level_fixed and next_level_fixed: reward = 1
    if (fixed and not next_fixed): reward = -1
    if level_fixed and not next_level_fixed: reward = -1

    return reward

def save_network(q_network, network_name):
    torch.save(q_network.state_dict(), network_name)
    print('Saved', network_name, ' to disk')

def update_q_network(state, next_state, action_idx, reward):
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

def update_target_network(target_network, q_network):
    global steps_done, target_update
    steps_done += 1
    if (steps_done % target_update == 0):
        target_network.load_state_dict(q_network.state_dict())


def execute_step(robot, data):
    global current_state, episode_score, episode_actions, episode_rewards, stale_state_counter, epsilon, epsilon_decay, neutral_actions
    action = get_action(robot, current_state)
    next_state = perform_action(robot, action)

    if (next_state != current_state or action in neutral_actions):
        stale_state_counter = 0
        robot.unlock()
        if (action in neutral_actions):
            reward = -0.1
        else:
            reward = get_reward(current_state, next_state)
        handle_reward(reward, next_state, data)
    else:
        stale_state_counter += 1

    if (stale_state_counter >= stale_state_limit):
        reward = -100
        handle_reward(reward, next_state, data)

def handle_reward(reward, next_state, data):
    global current_state, action_idx, target_network, q_network, episode_score, episode_actions, episode_rewards, epsilon, epsilon_decay, episodes_success
    next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_levelled, next_b_levelled = next_state

    # Step 1 : Ground fixed
    ground_fixed = (next_a_fixed and not next_a_levelled) or (next_b_fixed and not next_b_levelled)
    step1 = ground_fixed

    # Step 2 : Ground fixed and one module levelled
    step2 = step1 and (next_a_levelled or next_b_levelled)
    
    # Step 3 : Ground fixed and wall fixed
    wall_fixed = (next_a_fixed and next_a_levelled) or (next_b_fixed and next_b_levelled)
    step3 = wall_fixed and ground_fixed

    # Step 4 : Wall fixed and NOT ground fixed
    step4 = wall_fixed and not ground_fixed

    # Step 5 : Wall fixed and both modules levelled
    step5 = step3 and (next_a_levelled and next_b_levelled)

    # Step 6 : Both modules levelled and fixed
    step6 = next_a_levelled and next_b_levelled and next_a_fixed and next_b_fixed

    goal_condition = step6
    terminal_condition = next_a_distance >= 30 and next_b_distance >= 30

    update_q_network(current_state, next_state, action_idx, reward)
    update_target_network(target_network, q_network)
    current_state = next_state
    episode_score += reward
    episode_actions.append(action_space[action_idx])
    episode_rewards.append(reward)

    if (goal_condition): 
        episodes_success += 1
        reward = 100
    if (terminal_condition): reward = -100

    if (reward == 100 or reward == -100):
        end_episode(robot, data)

def end_episode(robot, data):
    global episode_counter, episode_score, current_state, highscore, episode_actions, episode_rewards, episode_start_time, stale_state_counter, simulation_model, acc_score, epsilon, epsilon_decay, episodes_success
    print("--------------------------------- END EPISODE ---------------------------------")
    print('Episode', episode_counter)
    print('State:', current_state)
    print('Score:', episode_score)
    print('Highscore:', highscore)
    print("success:", episodes_success)
    print('Episode actions:', episode_actions)
    print('Episode rewards:', episode_rewards)

    update_highscore(episode_score)
    episode_counter += 1
    episode_score = 0
    episode_actions.clear()
    episode_rewards.clear()
    episode_start_time = round(robot.get_data().time, 2)
    stale_state_counter = 0
    epsilon *= epsilon_decay
    current_state = initial_state
    reset_model()


def update_highscore(score):
    global highscore, highscore_actions, episode_actions
    if (score > highscore):
        highscore = score
        highscore_actions = episode_actions.copy()

#### MuJoCo Setup ####

robot_path = '../../model/index.xml'
robot_controller = controller

# Define the path to the XML model file.
xml_path = os.path.join(os.path.dirname(__file__), robot_path)

# Load the Mujoco model from the XML file.
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Set the control callback for the Mujoco model.
mj.set_mjcb_control(robot_controller)

glfw.init()
window = glfw.create_window(600, 450, "Lappa DQN", None, None)
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

# Set the initial value of sim_end, which determines when the simulation ends.
# A value of 0 means that the simulation will run indefinitely.
sim_end = 0

def reset_model():
    mj.mj_resetData(model, data)
    mj.mj_forward(model, data)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    robot.update(data)

while not glfw.window_should_close(window):
    sim_start = data.time
    target_fps = 60.0
    while (data.time - sim_start < 1.0/target_fps):
        mj.mj_step(model, data)

    # If sim_end is set and the simulation time has reached sim_end, exit the loop.
    if (sim_end > 0 and data.time >= sim_end or (episode_counter >= episodes)):
        print("--------------------------------- SIMULATION DONE ---------------------------------")
        print('Simulation done:', round(data.time, 2))
        print('Highscore:', highscore)
        print('Highscore actions:', highscore_actions)
        print('Episode successes:', episodes_success)
        save_network(q_network, network_name)
        break

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