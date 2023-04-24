import mujoco as mj
from mujoco.glfw import glfw
from deep_ql import controller
import os

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
window = glfw.create_window(600, 450, "Lappa", None, None)
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
