# MuJoCa Python API - mujoco_py
# This is a simple controller written in python for th Lappa model

from math import degrees

STATE = None
WALKSTEP = 0
TRANSITIONSTEP = 0
RULE = 2

def get_angle_degrees(joint_positions):
    return [degrees(pos) for pos in joint_positions]

def get_all_sensors_angle():
    global STATE
    sensors =  [get_angle_degrees(STATE.sensor('a_h1').data), 
                get_angle_degrees(STATE.sensor('b_h1').data),
                get_angle_degrees(STATE.sensor('a_h2').data),
                get_angle_degrees(STATE.sensor('b_h2').data),
                get_angle_degrees(STATE.sensor('a_h3').data),
                get_angle_degrees(STATE.sensor('b_h3').data), 
                STATE.sensor('a_rangefinder_forward').data, 
                STATE.sensor('b_rangefinder_forward').data,
                STATE.sensor('a_rangefinder_down').data,
                STATE.sensor('b_rangefinder_down').data]
    # round all values to 5 decimal places and flatmap the list
    return [round(item, 4) for sublist in sensors for item in sublist]

def is_grounded(module):
    global STATE
    if (module == "a"):
        return STATE.sensor("a_rangefinder_down").data < 0.01
    elif (module == "b"):
        return STATE.sensor("b_rangefinder_down").data < 0.01
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

# Align the modules so that both the a_h1 and b_h1 joints are at 0 degrees
# If the modules are not aligned, the module in front should rotate until they are
def align_modules():
    sensors = get_all_sensors_angle()
    if (sensors[0] > 0):
        set_rotate("a", -.1)
    elif (sensors[0] < 0):
        set_rotate("a", .1)
    else:
        set_rotate("a", 0)


def base_position():
    sensors = get_all_sensors_angle()
    set_rotate("b", 0)
    set_thrust("a", -1)
    set_thrust("b", .07)
    if (sensors[0] > 0):
        set_rotate("a", -.1)
    elif (sensors[0] < 0):
        set_rotate("a", .1)
    else:
        set_rotate("a", 0)
        set_thrust("b", -1)

def walk():
    global WALKSTEP
    sensors = get_all_sensors_angle()
    print(sensors[0], sensors[1])
    if (sensors[6] < 0.15 and sensors[7] < 0.15):
        set_thrust("a", -1)
        set_thrust("b", -1)
        return
    if (WALKSTEP == 0):
        if (sensors[6] < 0.15):
            return
        set_thrust("a", -1)
        set_thrust("b", .07)
        angle = sensors[0]
        if (angle > -30):
            set_rotate("a", -.3)
        else:
            set_rotate("a", 0)
            WALKSTEP = 1
    if (WALKSTEP == 1):
        if (sensors[7] < 0.15):
            return
        set_thrust("a", .07)
        set_thrust("b", -1)
        angle = sensors[1]
        if (angle < 30):
            set_rotate("b", .3)
        else:
            set_rotate("b", 0)
            WALKSTEP = 0

def transition():
    global STATE, TRANSITIONSTEP
    sensors = get_all_sensors_angle()

    if (TRANSITIONSTEP == 0):
        base_position()
        if (sensors[0] < .5 or sensors[1] > -.5):
            TRANSITIONSTEP = 1
    elif (TRANSITIONSTEP == 1):
        set_rotate("a", -.15)
        set_rotate("b", 0)
        set_thrust("a", -1.)
        set_thrust("b", .2)
        set_thrust("b", .075)

def print_state():
    global STATE
    a_rotate = STATE.actuator("a_rotor").ctrl
    b_rotate = STATE.actuator("b_rotor").ctrl
    a_thrust = STATE.actuator("a_thrust").ctrl
    b_thrust = STATE.actuator("b_thrust").ctrl
    print("a_rotate: ", a_rotate, "b_rotate: ", b_rotate, "a_thrust: ", a_thrust, "b_thrust: ", b_thrust)


def controller(model, data):
    global STATE, RULE, WALKSTEP, TRANSITIONSTEP

    b_h1 = data.joint('b_h1')
    b_h1.qpos = 0
    b_h1.qvel = 0
    b_h1.qacc = 0

    
    

    
    STATE = data
    sensors = get_all_sensors_angle()
        
    if is_grounded("a") and is_grounded("b"):
        if (RULE == 0):
            walk()
            #print(sensors[6], sensors[7])
        elif (RULE == 1):
            transition()
        elif (RULE == 2):
            #base_position()
            set_thrust("a", -1)
            set_thrust("b", .07)
            angle = sensors[0]
            if (angle > -30):
                set_rotate("a", -.3)
            else:
                set_rotate("a", 0)
                align_modules()
    print(sensors[0], sensors[1])

    pass
