import math
import sys
from api_v2 import *

API = None
DEGREE_OFFSET = 5


# print all actuator values and sensor values
def print_data(data):
    global API
    a_h1 = API.get_h1('a')
    a_h2 = API.get_h2("a")
    a_touch = API.get_touch("a")
    a_thrust = API.get_thruster("a")
    a_adhesion = API.get_adhesion("a")
    a_pressure = API.get_pressure("a")

    b_h1 = API.get_h1("b")
    b_h2 = API.get_h2("b")
    b_touch = API.get_touch("b")
    b_thrust = API.get_thruster("b")
    b_adhesion = API.get_adhesion("b")
    b_pressure = API.get_pressure("b")

    print("a_h1: " + str(a_h1) + " b_h1: " + str(b_h1))
    print("a_h2: " + str(a_h2) + " b_h2: " + str(b_h2))
    print("a_touch: " + str(a_touch) + " b_touch: " + str(b_touch))
    print("a_thrust: " + str(a_thrust) + " b_thrust: " + str(b_thrust))
    print("a_adhesion: " + str(a_adhesion) + " b_adhesion: " + str(b_adhesion))
    print("a_pressure: " + str(a_pressure) + " b_pressure: " + str(b_pressure))
    print("----------------------------------")


def fixate_module(module):
    global API
    API.set_thruster(module, -1)
    API.set_adhesion(module, 1)


def release_module(module):
    global API
    API.set_thruster(module, 0)
    API.set_adhesion(module, 0)


def lift_module(module):
    global API
    API.set_thruster(module, .85)
    API.set_adhesion(module, 0)


def lower_module(module):
    global API
    API.set_thruster(module, -.25)
    API.set_adhesion(module, .2)


def rotate_module(module, ctrl, angle):
    global API
    h1 = API.get_h1(module)
    diff = abs(h1 - angle)
    if (diff <= (DEGREE_OFFSET/2)):
        API.stop_rotation(module)
        return h1
    else:
        shortest_distance = min(diff, 360 - diff)
        if (h1 > angle and shortest_distance >= h1 - angle):
            API.rotate_module(module, -ctrl)
        else:
            API.rotate_module(module, ctrl)
        return h1


def get_angle_diff(angle_a, angle_b):
    diff = abs(angle_a - angle_b)
    return min(diff, 360 - diff)


def is_angle(module, angle):
    global API
    h1 = API.get_h1(module)
    diff = get_angle_diff(h1, angle)
    if (diff <= DEGREE_OFFSET):
        return True
    else:
        return False


def is_lifted(module):
    global API
    counter_module = 'a' if module == 'b' else 'b'
    h2 = API.get_h2(counter_module)
    pressure = API.get_pressure(module)
    if (h2 > 40 and pressure > -10):
        return True
    else:
        return False


def is_fixed(module):
    global API
    pressure = API.get_pressure(module)
    if (pressure < -90):
        return True
    else:
        return False


def get_state():
    # A state containg the following values:
    # 1. Whether the angle of the modules (h1 values) are set to 90 degrees
    # 2. Whether or not the modules are fixed
    # 3. Whether or not the modules are lifted

    global API
    a_fixed = is_fixed('a')
    b_fixed = is_fixed('b')
    a_lifted = is_lifted('a')
    b_lifted = is_lifted('b')
    a_h1 = is_angle("a", 90)
    b_h1 = is_angle("b", 90)
    return [a_fixed, b_fixed, a_lifted, b_lifted, a_h1, b_h1, ]


def jump():
    global API

    state = [is_fixed("a"), is_fixed("b"), is_lifted(
        "a"), is_lifted("b"), is_angle("a", 90), is_angle("b", 90)]
    print("state: ", state)

    step_0 = is_fixed("a") and is_fixed("b") and not is_angle(
        "a", 90) and not is_angle("b", 90)
    step_1 = not is_fixed("a") and is_fixed("b") and not is_angle(
        "a", 90) and not is_angle("b", 90) and not is_lifted("a")
    step_2 = not is_fixed("a") and is_fixed("b") and not is_angle(
        "a", 90) and not is_angle("b", 90) and is_lifted("a")
    step_3 = not is_fixed("a") and is_fixed("b") and not is_angle(
        "a", 90) and is_angle("b", 90) and is_lifted("a")
    step_4 = not is_fixed("a") and is_fixed("b") and not is_angle(
        "a", 90) and is_angle("b", 90) and not is_lifted("a")
    step_5 = is_fixed("a") and is_fixed("b") and not is_angle(
        "a", 90) and is_angle("b", 90) and not is_lifted("b")
    step_6 = is_fixed("a") and not is_fixed("b") and not is_lifted(
        "a") and not is_lifted("b") and not is_angle("a", 90)
    step_7 = is_fixed("a") and not is_fixed(
        "b") and is_lifted("b") and not is_angle("a", 90)
    step_8 = is_fixed("a") and not is_fixed(
        "b") and is_lifted("b") and is_angle("a", 90)
    step_9 = is_fixed("a") and not is_fixed(
        "b") and not is_lifted("b") and is_angle("a", 90)

    if (step_0):
        print("step_0")
    elif (step_1):
        lift_module("a")
        print("step_1")
    elif (step_2):
        rotate_module("b", .035, 90)
        print("step_2")
    elif (step_3):
        lower_module("a")
        print("step_3")
    elif (step_4):
        fixate_module("a")
        print("step_4")
    elif (step_5):
        release_module("b")
        print("step_5")
    elif (step_6):
        lift_module("b")
        print("step_6")
    elif (step_7):
        rotate_module("a", .035, 90)
        print("step_7")
    elif (step_8):
        lower_module("b")
        print("step_8")
    elif (step_9):
        fixate_module("b")
        print("step_9")
        return True
    else:
        print("Unknown state: ", [is_fixed("a"), is_fixed("b"), is_lifted("a"),
              is_lifted("b"), is_angle("a", 90), is_angle("b", 90)])
        sys.exit(0)
    return False

START = False
END = False

def controller(model, data):
    global API, START, END
    if (API is None):
        API = LappaApi(data)
        pass
    else:
        API.update_data(data)
        print_data(data)

    if (not START):
        fixate_module("b")
        fixate_module("a")

    if (is_fixed("b") and is_fixed("a") and not START):
        release_module("a")
        START = True
        return
    
    if (START and not END):
        jumped = jump()
        if (jumped):
            END = True
    
