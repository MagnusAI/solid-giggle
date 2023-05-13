import mujoco as mj
from mujoco.glfw import glfw
from controller import initialize, perform_action, neutral_actions, get_reward, end_episode, reset_simulation
from api import LappaApi
import os
import sys

ac = ['lower_b', 'rotate_b_forward', 'lower_b', 'lower_b', 'lift_b', 'rotate_b_backward', 'lower_b', 'lift_a', 'rotate_a_backward', 'lower_a', 'lift_a', 'lower_b', 'lower_b', 'lift_a', 'rotate_a_forward', 'lower_b', 'lower_b', 'rotate_b_forward', 'rotate_b_forward', 'stop_b_rotation', 'stop_a_rotation', 'rotate_a_forward', 'rotate_b_backward', 'stop_a_rotation', 'rotate_b_forward', 'lower_b', 'rotate_a_forward', 'lower_b', 'rotate_b_backward', 'lower_b', 'rotate_a_backward', 'rotate_a_backward', 'stop_a_rotation', 'lift_a', 'lift_b', 'rotate_a_forward', 'rotate_a_backward', 'lower_b', 'lower_b', 'lower_a', 'lower_b', 'lower_b', 'lower_b', 'rotate_b_forward', 'rotate_a_backward', 'lift_b', 'stop_b_rotation', 'stop_b_rotation', 'stop_b_rotation', 'rotate_b_forward', 'stop_b_rotation', 'rotate_a_backward', 'stop_b_rotation', 'rotate_b_backward', 'rotate_b_backward', 'lower_a', 'lower_b', 'rotate_b_backward']
action = None
debug_robot = None
current_state = None
score = 0
eee = 2
reset_delay = 0
rewards = []
actions_done = []
initial_data = None

def print_data_fields(data):
    for attr_name in dir(data):
        attr_value = getattr(data, attr_name)
        print(f'{attr_name}: {attr_value}')

def controller(model, data):
    global debug_robot, current_state, score, ac, action, eee, reset_delay, actions_done, rewards, initial_data
    if (debug_robot is None):
        initialize(model, data)
        debug_robot = LappaApi(data)
        initial_data = data
        current_state = debug_robot.read_state_from_sensors()
        print("State: ", current_state)
        ac.reverse()
        return
    else:
        debug_robot.update_data(data)

    if (reset_delay > 0):
        reset_delay -= 1
        current_state = debug_robot.read_state_from_sensors()
        if (reset_delay == 1):
            print("Reset State: ", current_state)
        return

    simulation_done = len(ac) == 0

    if (not simulation_done):
        if not debug_robot.is_locked(): action = ac.pop()
        next_state = perform_action(debug_robot, action)
        
        if (next_state != current_state or action in neutral_actions):
            print(action, next_state)
            debug_robot.unlock()
            if (action in neutral_actions):
                reward = -0.1
            else:
                reward = get_reward(current_state, next_state)
            
            next_a_fixed, next_b_fixed, next_arm_angle, next_a_distance, next_b_distance, next_a_leveled, next_b_leveled = next_state
            goal_condition = next_a_fixed or next_b_fixed and (next_a_leveled or next_b_leveled)
            terminal_condition = next_a_distance >= 30 and next_b_distance >= 30

            if (goal_condition): 
                reward = 100
            if (terminal_condition): reward = -100

            current_state = next_state
            score += reward
            actions_done.append(action)
            rewards.append(reward)

            if (reward == 100 or reward == -100):
                ac.clear()
    else:
        print("Simulation done")
        print("Episode score: ", score)
        print("Episode actions: ", actions_done)
        print("Episode rewards: ", rewards)
        print("-------------------------------------")
        if (eee > 0):
            score = 0
            ac.clear()
            debug_robot.unlock()
            actions_done.clear()
            rewards.clear()
            ac = ['lower_b', 'lift_a', 'lift_a', 'stop_b_rotation', 'lower_a', 'lower_a']
            ac.reverse()
            reset_simulation(model, data, debug_robot)
            current_state = debug_robot.read_state_from_sensors()
            print("State: ", current_state)
            reset_delay = 1000
            eee -= 1
        else:
            sys.exit(0)

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
