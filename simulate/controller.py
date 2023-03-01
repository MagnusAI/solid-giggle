# MuJoCa Python API - mujoco_py
# This is a simple controller written in python for th Lappa model

from math import degrees

STATE = None

# SENSOR THRESHOLDS
GROUND_THRESHOLD = 0.01
OBSTACLE_THRESHOLD = 0.15

# convert scalar to degrees
def deg(scalar):
    return round(degrees(scalar), 2)

def get_sensor_values():
    global STATE
    sensors =  [STATE.sensor('a_h1').data, 
                STATE.sensor('b_h1').data,
                STATE.sensor('a_h2').data,
                STATE.sensor('b_h2').data,
                STATE.sensor('a_h3').data,
                STATE.sensor('b_h3').data, 
                STATE.sensor('a_rangefinder_forward').data, 
                STATE.sensor('b_rangefinder_forward').data,
                STATE.sensor('a_rangefinder_down').data,
                STATE.sensor('b_rangefinder_down').data]
    # round all values to 4 decimal places and flatmap the list
    return [round(item, 4) for sublist in sensors for item in sublist]

def is_grounded(module):
    global STATE, GROUND_THRESHOLD
    if (module == "a"):
        return STATE.sensor("a_rangefinder_down").data < GROUND_THRESHOLD
    elif (module == "b"):
        return STATE.sensor("b_rangefinder_down").data < GROUND_THRESHOLD
    else:
        print(is_grounded.__name__ + ": Invalid module name")

def set_thrust(module, thrust):
    global STATE
    if (module == "a"):
        STATE.actuator("a_thrust").ctrl = thrust
    elif (module == "b"):
        STATE.actuator("b_thrust").ctrl = thrust
    else:
        print(set_thrust.__name__ + ": Invalid module name")

def set_rotate(module, ctrl):
    global STATE
    if (module == "a"):
        STATE.actuator("a_rotor").ctrl = ctrl
    elif (module == "b"):
        STATE.actuator("b_rotor").ctrl = ctrl
    else:
        print(set_rotate.__name__ + ": Invalid module name")

def release_module(module):
    global STATE
    if (module == "a"):
        STATE.actuator("a_thrust").ctrl = .01
    elif (module == "b"):
        STATE.actuator("a_thrust").ctrl = .01
    else:
        print(release_module.__name__ + ": Invalid module name")

def is_obstructed(module):
    global STATE, OBSTACLE_THRESHOLD
    if (module == "a"):
        return STATE.sensor("a_rangefinder_forward").data < OBSTACLE_THRESHOLD
    elif (module == "b"):
        return STATE.sensor("b_rangefinder_forward").data < OBSTACLE_THRESHOLD
    else:
        print(is_obstructed.__name__ + ": Invalid module name")

def print_actuators():
    global STATE
    a_rotate = STATE.actuator("a_rotor").ctrl
    b_rotate = STATE.actuator("b_rotor").ctrl
    a_thrust = STATE.actuator("a_thrust").ctrl
    b_thrust = STATE.actuator("b_thrust").ctrl
    print("a_rotate: ", a_rotate, "b_rotate: ", b_rotate, "a_thrust: ", a_thrust, "b_thrust: ", b_thrust)

def print_sensors():
    sensors = get_sensor_values()
    print("a_h1: ", sensors[0], "b_h1: ", sensors[1], 
          "\na_h2: ", sensors[2], "b_h2: ", sensors[3], 
          "\na_h3: ", sensors[4], "b_h3: ", sensors[5], 
          "\na_rangefinder_forward: ", sensors[6], "b_rangefinder_forward: ", sensors[7], 
          "\na_rangefinder_down: ", sensors[8], "b_rangefinder_down: ", sensors[9])

def controller(model, data):
    global STATE
    STATE = data
    print(deg(data.sensor('a_h1').data), deg(data.sensor('b_h1').data))
    if (is_grounded("a") and is_grounded("b")):
        set_thrust("a", -1)
        set_thrust("b", 0)
        a_angle = get_sensor_values()[0]
        if (deg(a_angle) > -90.):
            set_rotate("a", -.75)
        else:
            set_rotate("a", 0)
            set_thrust("b", -1)
            set_thrust("a", 0)


    pass
