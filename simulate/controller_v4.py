import os
import sys
from api_v2 import *

API = None
END = False
STATE = None

# A dictionary that maps a state to a rule
RULES = {
    (True, True, False, False, False): "s0",
    (True, False, False, False, False): "s1",
    (True, False, True, False, False): "s2",
    (True, False, True, True, False): "s3",
    (True, True, True, True, False): "s4",
    (False, True, True, True, False): "s5",
    (False, True, True, False, False): "s6",
    (False, True, False, False, False): "s7",
    (False, True, False, True, False): "s8",
    (True, True, False, True, False): "s9",
    (True, False, False, True, False): "s10",
    (True, True, True, False, False): "s11",
    (False, False, False, True, False): "s12",
    (False, False, True, True, False): "s13",
    (False, False, True, False, False): "s14",
    (False, False, False, False, False): "s15",

    (True, True, False, False, True): "s16",
    (False, False, False, True, True): "s12",

    (True, False, False, False, True): "s1",
    (True, False, True, False, True): "s2",
    (True, False, True, True, True): "s3",
    (True, True, True, True, True): "s4",
    (False, True, True, True, True): "s5",
    (False, True, True, False, True): "s6",
    (False, True, False, False, True): "s7",
    (False, True, False, True, True): "s8",
    (True, True, False, True, True): "s9",
    (True, False, False, True, True): "s10",
    (True, True, True, False, True): "s11",
    (False, False, True, True, True): "s13",
    (False, False, True, False, True): "s14",
    (False, False, False, False, True): "s15",
}


def print_debug_data(data):
    global API
    a_pos = API.get_position("a")
    b_pos = API.get_position("b")

    a_h1 = API.get_h1('a')
    a_h2 = API.get_h2("a")
    a_touch = API.get_touch("a")
    a_thrust = API.get_thruster("a")
    a_h1_thrust = API.get_h1_actuator("a")
    a_adhesion = API.get_adhesion("a")
    a_pressure = API.get_pressure("a")

    b_h1 = API.get_h1("b")
    b_h2 = API.get_h2("b")
    b_touch = API.get_touch("b")
    b_thrust = API.get_thruster("b")
    b_h1_thrust = API.get_h1_actuator("b")
    b_adhesion = API.get_adhesion("b")
    b_pressure = API.get_pressure("b")

    print("----------------------------------")
    print("position: " + str(a_pos) + " | " + str(b_pos))
    print("a_h1: " + str(a_h1) + " b_h1: " + str(b_h1))
    print("a_h2: " + str(a_h2) + " b_h2: " + str(b_h2))
    print("a_touch: " + str(a_touch) + " b_touch: " + str(b_touch))
    print("a_thrust: " + str(a_thrust) + " b_thrust: " + str(b_thrust))
    print("a_h1_thrust: " + str(a_h1_thrust) +
          " b_h1_thrust: " + str(b_h1_thrust))
    print("a_adhesion: " + str(a_adhesion) + " b_adhesion: " + str(b_adhesion))
    print("a_pressure: " + str(a_pressure) + " b_pressure: " + str(b_pressure))
    print("----------------------------------")


def lower(module):
    global API
    API.set_thruster(module, -.5)
    pressure = API.get_pressure(module)
    if (pressure < 0):
        API.set_thruster(module, -1)
        API.set_adhesion(module, 1)


def lift(module):
    global API
    API.set_adhesion(module, 0)
    API.set_thruster(module, .5)


def rotate(module, ctrl):
    global API
    counter_module = "b" if module == "a" else "a"
    API.stop_rotation(counter_module)
    API.rotate_module(module, ctrl)


def get_target_angle(module):
    global API
    h1 = API.get_h1(module)

    direction = 90 if h1 < 90 else -90
    target = h1 + direction
    return target % 360 if target > 0 else target + 360


def is_angle(module, target):
    global API
    h1 = round(API.get_h1(module), 1)
    d1 = abs(target - h1)
    d2 = 360 - d1
    diff = min(d1, d2)
    allowed_offset = 5
    return diff < allowed_offset


def rotate_to(module, target, ctrl=.5):
    global API
    if (is_angle(module, target)):
        API.stop_rotation(module)
        return
    else:
        API.set_thruster(module, .1)
        h1 = round(API.get_h1(module), 1)
        diff = abs(h1 - target) % 360
        if diff <= 180:
            ctrl = ctrl if target > h1 else -ctrl
        else:
            ctrl = ctrl if target < h1 else -ctrl
        rotate(module, ctrl)


def is_lifted():
    global API
    a_h2 = API.get_h2("a")
    b_h2 = API.get_h2("b")
    acc = a_h2 + b_h2
    offset = 5
    return acc > (45 - offset)


def is_rotated():
    one_90 = is_angle("a", 90) or is_angle("b", 90)
    both_90 = is_angle("a", 90) and is_angle("b", 90)
    one_270 = is_angle("a", 270) or is_angle("b", 270)
    both_270 = is_angle("a", 270) and is_angle("b", 270)
    rot_90 = one_90 and not both_90
    rot_270 = one_270 and not both_270
    return rot_90 or rot_270


def is_leveled():
    global API
    a_pos = round(API.get_position("a")[2], 2)
    b_pos = round(API.get_position("b")[2], 2)
    return a_pos > 0.1 and b_pos > 0.1


def get_state():
    global STATE
    a_fixed = API.get_pressure("a") < -100
    b_fixed = API.get_pressure("b") < -100
    lifted = is_lifted()
    rotated = is_rotated()
    leveled = is_leveled()
    state = (a_fixed, b_fixed, lifted, rotated, leveled)

    if (state != STATE):
        # Debug
        # print_debug_data(state)
        # print("state: " + str(state))
        # print("rule: " + get_rule(state))
        STATE = state

    return state


def get_rule(state):
    global RULES
    return RULES[state]


def perform_action(rule):
    global END
    if (rule == "s0"):
        lift("b")
    elif (rule == "s1"):
        lift("b")
    elif (rule == "s2"):
        rotate_to("a", 270)
    elif (rule == "s3"):
        lower("b")
    elif (rule == "s4"):
        lift("a")
    elif (rule == "s5"):
        rotate_to("b", 80)
    elif (rule == "s6"):
        lower("a")
    elif (rule == "s7"):
        lower("a")
    elif (rule == "s8"):
        rotate_to("b", 80)
    elif (rule == "s9"):
        lift("a")
    elif (rule == "s10"):
        rotate_to("a", 270)
    elif (rule == "s11"):
        lift("b")
    elif (rule == "s16"):
        END = True
    else:
        lower("a")
        lower("b")


def controller(model, data):
    global API, END

    if (END):
        return

    if (API is None):
        API = LappaApi(data)
        return
    else:
        API.update_data(data)
        # print_debug_data(data)

    state = get_state()
    rule = get_rule(state)
    perform_action(rule)

    if (data.time > 30):
        END = True

    if (END):
        state = get_state()
        result = {
            "state": state,
            "a_pos": API.get_position("a"),
            "b_pos": API.get_position("b"),
            "success": is_leveled() and state[0] and state[1]
        }

        with open("output/results.txt", "a") as f:
            f.write(str(result) + "\n")

        sys.exit(0)
