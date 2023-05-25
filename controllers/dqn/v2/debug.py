import mujoco as mj
from mujoco.glfw import glfw
from controller import initialize, perform_action, neutral_actions, get_reward, end_episode, reset_simulation
from api import LappaApi
import os
import sys


debug_robot = None
initial_data = None
initial_state = None
current_state = None
actions_done = []
todo = ['lower_b', 'lower_b', 'lift_a']
debug_actions = todo.copy()
episodes = 2
action = None
stale_count = 0
stale_limit = 3000
rewards = []
score = 0


def print_data_fields(data):
    for attr_name in dir(data):
        attr_value = getattr(data, attr_name)
        print(f'{attr_name}: {attr_value}')

def controller(model, data):
    global debug_robot, initial_data, initial_state, current_state, actions_done, todo, debug_actions, episodes, action, stale_count, stale_limit, rewards, score
    if debug_robot is None:
        debug_robot = LappaApi(data)
        data = reset_simulation(data)
        initial_state = debug_robot.get_state()
        current_state = initial_state
        initial_data = data
        debug_actions.reverse()
    else:
        debug_robot.update_data(data)

    simulation_done = len(actions_done) == len(todo)
    if (episodes < 1):
        sys.exit(0)

    if (not simulation_done):
        if (not debug_robot.is_locked()):
            action = debug_actions.pop()
        next_state = perform_action(debug_robot, action)

        if (next_state != current_state):
            stale_count = 0
            debug_robot.unlock()
            if (action in neutral_actions):
                reward = -0.1
            else:
                reward = get_reward(current_state, next_state)
            
            print("Reward: ", reward, "Action: ", action, "State: ", current_state, "Next State: ", next_state)
            actions_done.append(action)
            rewards.append(reward)
            score += reward
            current_state = next_state
        else: 
            stale_count += 1
        
        if (stale_count > stale_limit):
            reward = -100
            print("Stale count exceeded stale limit. Reward: ", reward, "Action: ", action, "State: ", current_state, "Next State: ", next_state)
            actions_done.append(action)
            rewards.append(reward)
            score += reward
    else:
        print("--------------- Episode:", episodes, "--------------------")
        print("Simulation done. Score: ", score)
        print("Actions done: ", actions_done)
        print("Rewards: ", rewards)
        print("-----------------------------------")
        score = 0
        data = reset_simulation(initial_data)
        debug_actions = todo.copy()
        debug_actions.reverse()
        actions_done = []
        rewards = []
        episodes -= 1
    
robot_path = '../../../model/index.xml'
robot_controller = controller

# Define the path to the XML model file.
xml_path = os.path.join(os.path.dirname(__file__), robot_path)

# Load the Mujoco model from the XML file.
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Set the control callback for the Mujoco model.
mj.set_mjcb_control(robot_controller)

glfw.init()
window = glfw.create_window(600, 450, "DEBUG", None, None)
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

while not glfw.window_should_close(window):
    sim_start = data.time
    target_fps = 60.0
    while (data.time - sim_start < 1.0/target_fps):
        mj.mj_step(model, data)

    # If sim_end is set and the simulation time has reached sim_end, exit the loop.
    if (sim_end > 0 and data.time >= sim_end):
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
