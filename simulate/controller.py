import math
from api import *

API = None
STATE = None
MOVE_TARGET = None
MOVE_STEP = 0

# The robot state is represented by an array of 4 elements:
# [af, bf, ar, br]
# Where ar and br are the modules rangefinder detecting obstacles
# and af and bf are the modules force sensors detecting the ground
# the state use booleans to represent the presence of an obstacle
# or the presence of the ground

# RULES define the action to take when a state is detected
RULES = {
    # No obstacle avoidance for now, we just try to move past/through
    (True, True, False, False): "move",
    # No obstacle avoidance for now, we just try to move past/through
    (True, True, False, True): "move",
    (True, True, True, False): "move",
    (True, True, True, True): "climb",
    (False, True, True, True): "climb",
    (True, False, True, True): "climb",
    (False, True, False, True): "jump",
    (True, False, True, False): "jump",
    (False, True, True, False): "descend",
    (False, False, True, True): "descend",
    (True, False, False, True): "descend",
    (False, True, False, False): "descend",
    (True, False, False, False): "descend",
    (False, False, False, True): "descend",
    (False, False, True, False): "descend",
    (False, False, False, False): "descend",
}


def get_state():
    global API
    # Read the sensors
    ar = False #API.read_sensors("a", ["range"])[0]
    br = False #API.read_sensors("b", ["range"])[0]
    af = API.read_sensors("a", ["vacuum"])[0]
    bf = API.read_sensors("b", ["vacuum"])[0]

    # Convert the sensor values to booleans
    af = af < -1
    #ar = ar < 0.1
    bf = bf < -1
    #br = br < 0.1

    # Return the state
    return (af, bf, ar, br)


def get_action(state):
    # Get the action to take from the rules
    action = RULES[state]

    # Return the action
    return action


def move():
    global MOVE_STEP
    if (MOVE_STEP == 0) :
        if (forward_module("a", 45)):
            MOVE_STEP = 1
    elif (MOVE_STEP == 1):
        if (forward_module("b", -90)):
            MOVE_STEP = 2
    elif (MOVE_STEP == 2):
        if (forward_module("a", 90)):
            MOVE_STEP = 1
    else: 
        pass


def forward_module(module, degrees):
    global API, MOVE_TARGET

    counter_module = "a" if module == "b" else "b"
    API.lock()
    h1 = API.read_sensors(counter_module, ["h1"])[0]
    h1_deg = math.degrees(h1) % 360

    if (MOVE_TARGET is None):
        MOVE_TARGET = (h1_deg + degrees) % 360

    diff = abs(MOVE_TARGET - h1_deg) % 360
    min_diff = min(diff, 360 - diff)


    if (min_diff > 5):
        API.fix_module(counter_module)
        API.release_module(module)
        API.rotate_module(counter_module, -.25)
    else:
        API.set_actuator(counter_module, "h1", 0)
        API.fix_module(module)
        MOVE_TARGET = None
        API.unlock()
        return True
    return False



def climb():
    # TODO: Implement this
    pass


def jump():
    # TODO: Implement this
    pass


def descend():
    global API
    sensor_a = API.read_sensors("a", ["vacuum"])[0]
    sensor_b = API.read_sensors("b", ["vacuum"])[0]
    floating_value = -.3
    if sensor_a < floating_value and sensor_b < floating_value:
        API.fix_module("a")
        API.fix_module("b")
    elif sensor_a < floating_value:
        API.fix_module("a")
        API.set_actuator("b", "thrust", -1)
    elif sensor_b < floating_value:
        API.fix_module("b")
        API.set_actuator("a", "thrust", -1)
    else:
        API.set_actuator("a", "thrust", -1)
        API.set_actuator("b", "thrust", -1)
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


TEST = True
START = False


def controller(model, data):
    global API, STATE, TEST, START
    if (API is None):
        API = LappaApi(data)
        pass
    else:
        API.update_data(data)

    if (not API.is_locked()):
        STATE = get_state()

    if (STATE == (True, True, False, False)):
        START = True

    if (START):
        move()
        

    # Get the action to take
    #action = get_action(STATE)
    # Perform the action
    # perform_action(action)

    pass
