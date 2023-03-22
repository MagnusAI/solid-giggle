import mujoco as mj
from mujoco.glfw import glfw
from controller import controller
import os

# Set path to XML file
xml_path = os.path.join(os.path.dirname(__file__), '../model/lappa.xml')

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)
data = mj.MjData(model)

# Set controller
mj.set_mjcb_control(controller)

# Set up GLFW window and mouse/keyboard callbacks
glfw.init()
window = glfw.create_window(1200, 900, "Lappa", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# Initialize visualization data structures
cam = mj.MjvCamera()
opt = mj.MjvOption()
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)

# Simulation loop
while not glfw.window_should_close(window):
    # Step simulation
    simstart = data.time
    while (data.time - simstart < 1.0/60.0):
        mj.mj_step(model, data)

    # Update and render visualization
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # Swap buffers and process events
    glfw.swap_buffers(window)
    glfw.poll_events()

# Clean up
glfw.terminate()
