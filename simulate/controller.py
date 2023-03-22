from api import *

API = None

# The robot state is represented by an array of 4 elements:
# [ar, br, af, bf]
# Where ar and br are the modules rangefinder detecting obstacles
# and af and bf are the modules force sensors detecting the ground
# the state use booleans to represent the presence of an obstacle
# or the presence of the ground

STATE = (False, False, False, False)

# RULES define the action to take when a state is detected
RULES = {
    # If the robot is in the air, it will try to land
    (False, False, False, False): "land",
    # If it is on the ground and it is not obstructed, it will try to move
    (True, True, False, False): "move",
    # If it is on the ground and it is obstructed, it will try climb up
    (True, True, True, False): "climb",
    (True, True, False, True): "climb",
    (True, True, True, True): "climb",
    # If it is on the ground with one module and not the other, and the grounded module is obstructed, it will try to jump
    (True, False, True, False): "jump",
    (False, True, False, True): "jump",
    # If it is on the ground with one module and not the other, it will try to descend
    (True, False, False, False): "descend",
    (False, True, False, False): "descend",
}

def get_state(API, data):
    # Read the sensors
    ar = API.read_sensors("a", ["range"])[0]
    br = API.read_sensors("b", ["range"])[0]
    af = API.read_sensors("a", ["vacuum"])[0]
    bf = API.read_sensors("b", ["vacuum"])[0]
    
    # Convert the sensor values to booleans
    ar = ar > 0.1
    br = br > 0.1
    af = af > 0.1
    bf = bf > 0.1
    
    # Return the state
    return (ar, br, af, bf)

def get_action(API, state):
    # Get the action to take from the rules
    action = RULES[state]
    
    # Return the action
    return action

def land(API, data):    
    # Set the thrust of the modules to -1
    API.set_actuator("a", "thrust", -1)
    API.set_actuator("b", "thrust", -1)

def move(API, data):
    # TODO: Implement this
    pass

def climb(API, data):
    # TODO: Implement this
    pass

def jump(API, data):
    # TODO: Implement this
    pass

def descend(API, data):
    # TODO: Implement this
    pass

def perform_action(API, action, data):
    # Perform the action
    if action == "land":
        land(API, data)
    elif action == "move":
        move(API, data)
    elif action == "climb":
        climb(API, data)
    elif action == "jump":
        jump(API, data)
    elif action == "descend":
        descend(API, data)
    else:
        raise Exception("Unknown action: " + action)

def controller(model, data):
    global API
    if (API is None):
        API = LappaApi(data)
        pass
    else:
        API.update_state(data)

    print(data.sensor("a_range").data[0])
    
    # Get the state
    state = get_state(API, data)
    # Get the action to take
    action = get_action(API, state)
    # Perform the action
    perform_action(API, action, data)

    pass
