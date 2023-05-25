import os
import sys
from api_v2 import *

API = None
END = False
STATE = None
AXIS = 2  # 0 = x, 1 = y, 2 = z

# A dictionary that maps a state to a rule
# RULES = {
#     (True, True, False, False, False): "r0",
#     (True, False, False, False, False): "r1",
#     (True, False, True, False, False): "r2",
#     (True, False, True, True, False): "r3",
#     (True, True, True, True, False): "r4",
#     (False, True, True, True, False): "r5",
#     (False, True, True, False, False): "r5",
#     (False, True, False, False, False): "r7",
#     (False, True, False, True, False): "r8",
#     (True, True, False, True, False): "r9",
#     (True, False, False, True, False): "r10",
#     (True, True, True, False, False): "r11",
#     (False, False, False, True, False): "r12",
#     (False, False, True, True, False): "r13",
#     (False, False, True, False, False): "r14",
#     (False, False, False, False, False): "r15",

#     (True, True, False, False, True): "r16",
#     (False, False, False, True, True): "r12",

#     (True, False, False, False, True): "r1",
#     (True, False, True, False, True): "r2",
#     (True, False, True, True, True): "r3",
#     (True, True, True, True, True): "r4",
#     (False, True, True, True, True): "r6",
#     (False, True, True, False, True): "r5",
#     (False, True, False, False, True): "r7",
#     (False, True, False, True, True): "r7",
#     (True, True, False, True, True): "r9",
#     (True, False, False, True, True): "r10",
#     (True, True, True, False, True): "r11",
#     (False, False, True, True, True): "r13",
#     (False, False, True, False, True): "r14",
#     (False, False, False, False, True): "r15",
# }

RULES = {
    (False, False, False, False, False): "r0",
    (False, False, False, False, True): "r0",
    (False, False, False, True, False): "r0",
    (False, False, False, True, True): "r0",
    (False, False, True, False, False): "r0",
    (False, False, True, False, True): "r0",
    (False, False, True, True, False): "r0",
    (False, False, True, True, True): "r0",

    (True, True, False, False, False): "r1",
    (True, True, False, False, True): "goal",
    (True, True, False, True, False): "r1",
    (True, True, False, True, True): "goal",
    (True, True, True, False, False): "r2",
    (True, True, True, False, True): "goal",
    (True, True, True, True, False): "r2",
    (True, True, True, True, True): "goal",

    (True, False, False, False, False): "r1",
    (True, False, False, False, True): "r0",
    (True, False, False, True, False): "r3",
    (True, False, False, True, True): "r0",
    (True, False, True, False, False): "r4",
    (True, False, True, False, True): "r0",
    (True, False, True, True, False): "r0",
    (True, False, True, True, True): "r0",

    (False, True, False, False, False): "r0",
    (False, True, False, False, True): "r0",
    (False, True, False, True, False): "r0",
    (False, True, False, True, True): "r0",
    (False, True, True, False, False): "r0",
    (False, True, True, False, True): "r0",
    (False, True, True, True, False): "r5",
    (False, True, True, True, True): "r0",

}


def get_state():
    global STATE
    a_fixed = API.get_pressure("a") < -45
    b_fixed = API.get_pressure("b") < -45
    lifted = is_lifted()
    rotated = is_rotated()
    levelled = is_levelled()
    state = (a_fixed, b_fixed, lifted, rotated, levelled)

    if (state != STATE):
        # Debug
        print_debug_data()
        print("state: " + str(state))
        print("rule: " + get_rule(state))
        STATE = state

    return state


def perform_action(rule):
    global END
    if (rule == "r0"):
        lower("a")
        lower("b")
    elif (rule == "r1"):
        lift("b")
    elif (rule == "r2"):
        lift("a")
    elif (rule == "r3"):
        rotate("a", 270)
    elif (rule == "r4"):
        rotate("a", 90)
    elif (rule == "r5"):
        rotate("b", 90)
    elif (rule == "goal"):
        stop()
    else:
        print("No action")
        stop()

# def perform_action(rule):
#     global END
#     if (rule == "r0"):
#         lift("b")
#     elif (rule == "r1"):
#         lift("b")
#     elif (rule == "r2"):
#         rotate_to("a", 270)
#     elif (rule == "r3"):
#         lower("b")
#     elif (rule == "r4"):
#         lift("a")
#     elif (rule == "r5"):
#         rotate_to("b", 80)
#     elif (rule == "r6"):
#         lower("a")
#     elif (rule == "r7"):
#         lower("a")
#     elif (rule == "r8"):
#         lower("a")
#     elif (rule == "r9"):
#         lift("a")
#     elif (rule == "r10"):
#         rotate_to("a", 270)
#     elif (rule == "r11"):
#         lift("b")
#     elif (rule == "r12"):
#         lower("b")
#     elif (rule == "r13"):
#         lower("b")
#     elif (rule == "r16"):
#         stop()
#     else:
#         lower("a")
#         lower("b")


def print_debug_data():
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
    API.set_thruster(module, -1)
    pressure = API.get_pressure(module)
    if (pressure < -2):
        API.set_adhesion(module, .55)


def lift(module):
    global API
    API.set_adhesion(module, 0)
    API.set_thruster(module, .2)
    API.reset_module(module)


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


def is_levelled():
    global API, AXIS
    a_pos = round(API.get_position("a")[AXIS], 2)
    b_pos = round(API.get_position("b")[AXIS], 2)
    return a_pos > 0.15 and b_pos > 0.15


def get_rule(state):
    global RULES
    return RULES[state]


def stop():
    global END
    END = True


ACTIONS = []


def controller(model, data):
    global API, END, ACTIONS

    if (API is None):
        API = LappaApi(data)
        return
    else:
        API.update_data(data)

    state = get_state()
    rule = get_rule(state)
    perform_action(rule)

    if (len(ACTIONS) == 0 or ACTIONS[-1] != rule):
        ACTIONS.append(rule)
        if (len(ACTIONS) > 100):
            stop()

    if (data.time > 60):
        stop()

    if (END):
        state = get_state()
        result = {
            "state": state,
            "a_pos": API.get_position("a"),
            "b_pos": API.get_position("b"),
            "actions_count": len(ACTIONS),
            "time": data.time,
            "success": is_levelled() and state[0] and state[1]
        }

        with open("output/results.txt", "a") as f:
            f.write(str(result) + "\n")

        print("Actions: ", ACTIONS)
        sys.exit(0)
