import os
import sys
from api_v3 import *

API = None
ACTIONS = []
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward', 'stop_a_rotation', 'stop_b_rotation']


def get_h2(data, module):
    scalar = data.sensor(module + "_h2").data[0]
    degrees = math.degrees(scalar)
    if (module == "b"):
        degrees = -degrees
    return degrees


def debug_debug(data):
    print("------------------- DATA --------------------------------------")
    print("H1: ", round(data.sensor("a_h1").data[0], 1),
          " , ", round(data.sensor("b_h1").data[0], 1))
    print("H2: ", round(get_h2(data, "a"), 1),
          " , ", round(get_h2(data, "b"), 1))
    print("Pressure: ", round(data.sensor("a_vacuum").data[2], 1),
          " , ", round(data.sensor("b_vacuum").data[2], 1))
    print("Thrust: ", data.actuator(
        "a_thrust").ctrl[0], " , ", data.actuator("b_thrust").ctrl[0])
    print("Rotation: ", round(data.actuator("a_h1").ctrl[0], 1),
          " , ", round(data.actuator("b_h1").ctrl[0], 1))


def controller(model, data):
    global API, ACTIONS

    if (API is None):
        API = LappaApi(data)
        return
    else:
        API.update_data(data)
        API.debug_info()
        #debug_debug(data)

    state = API.read_state_from_sensors()
    init_condition = not state[0] and not state[1] and not state[2] and not state[5] and not state[6]
    one_condition = state[0] and state[1]
    two_condition = state[0] and state[2]
    three_condition = state[0] and state[2] and state[4] < 10 and state[4] > -5

    if (init_condition):
        API.perform_action("lower_a")
        API.perform_action("lower_b")
        return
    if (one_condition):
        API.perform_action("lift_b")
        return
    if (two_condition):
        API.perform_action("rotate_a_backward")
        pass
    if (three_condition):
        API.perform_action("stop_a_rotation")
        pass
