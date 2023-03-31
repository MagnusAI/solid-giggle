import math
import sys
from api import *

API = None
STATE = None
MOVE_TARGET = None
MOVE_STEP = 0
JUMP_STEP = 0

def lift_module(module):
    global API
    counter_module = "a" if module == "b" else "b"
    API.fix_module(counter_module)
    API.stop_rotation(counter_module)
    API.set_actuator(module, "vacuum", 0)
    API.set_actuator(module, "thrust", .8)
    a_h2 = math.degrees(API.read_sensors("a", ["h2"])[0])
    b_h2 = math.degrees(API.read_sensors("b", ["h2"])[0])
    if (abs(a_h2) > 35 and abs(b_h2) > 35):
        return True
    return False

def lower_module(module):
    global API
    API.set_actuator(module, "thrust", -.5)
    API.set_actuator(module, "vacuum", .3)
    surfaced = API.read_sensors(module, ["vacuum"])[0] < 0
    if (surfaced):
        return True
    return False

def jump():
    global API, JUMP_STEP
    if (JUMP_STEP == 0):
        step_complete = lift_module("a")
        if (step_complete):
            JUMP_STEP = 1
    elif (JUMP_STEP == 1):
        step_complete = angle_module("b", 90)
        if (step_complete):
            JUMP_STEP = 2
    elif (JUMP_STEP == 2):
        step_complete = lower_module("a")
        if (step_complete):
            JUMP_STEP = 3
    elif (JUMP_STEP == 3):
        API.set_actuator("a", "thrust", .3)
        API.set_actuator("b", "thrust", .3)
        step_complete = reset_angle("a") and reset_angle("b")
        if (step_complete):
            API.set_actuator("a", "thrust", 0.6)
            if (angle_module("a", -350)):
                API.fix_module("a")
                API.stop_rotation("a")
                API.stop_rotation("b")
                JUMP_STEP = 4
    elif (JUMP_STEP == 4):
        step_complete = lift_module("b")
        if (step_complete):
            JUMP_STEP = 5
    elif (JUMP_STEP == 5):
        pass

    return False




def move():
    global MOVE_STEP
    a_touch = API.read_sensors("a", ["touch"])[0] > 0
    b_touch = API.read_sensors("b", ["touch"])[0] > 0
    if (a_touch or b_touch):
        API.unlock()
        pass

    if (MOVE_STEP == 0) :
        API.fix_module("a")
        API.release_module("b")
        if (angle_module("a", -45)):
            API.fix_module("b")
            MOVE_STEP = 1
    elif (MOVE_STEP == 1):
        API.fix_module("b")
        API.release_module("a")
        if (angle_module("b", 90)):
            API.fix_module("a")
            MOVE_STEP = 2
    elif (MOVE_STEP == 2):
        API.fix_module("a")
        API.release_module("b")
        if (angle_module("a", -90)):
            API.fix_module("b")
            MOVE_STEP = 1
    else: 
        pass


def angle_module(module, degrees):
    global API, MOVE_TARGET

    h1 = API.read_sensors(module, ["h1"])[0]
    h1_deg = math.degrees(h1) % 360
    print("h1_deg: " + str(h1_deg))

    if (MOVE_TARGET is None):
        MOVE_TARGET = (h1_deg + degrees) % 360

    diff = abs(MOVE_TARGET - h1_deg) % 360
    min_diff = min(diff, 360 - diff)

    if (min_diff > 5):
        API.fix_module(module)
        API.rotate_module(module, -.75)
    else:
        API.stop_rotation(module)
        MOVE_TARGET = None
        return True
    return False



def climb():
    # TODO: Implement this
    pass

def descend():
    pass


def perform_action(action):
    # Perform the action
    if action == "move":
        move()
    elif action == "climb":
        climb()
    elif action == "jump":
        jump()
    elif action == "descend":
        descend()
    else:
        raise Exception("Unknown action: " + action)

def reset_angle(module):
    global API
    if (angle_module(module, 0)):
        return True
    return False

TEST = False
STEP = 0

def controller(model, data):
    global API, STATE, TEST, STEP
    if (API is None):
        API = LappaApi(data)
        pass
    else:
        API.update_data(data)

    a_vaccum = round(API.read_sensors("a", ["vacuum"])[0])
    b_vaccum = round(API.read_sensors("b", ["vacuum"])[0])

    landed = a_vaccum < 0 and b_vaccum < 0
    if (landed):
        TEST = True
    
    if (TEST):
        jump()
        print(JUMP_STEP)
    pass
