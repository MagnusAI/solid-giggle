# MuJoCa Python API - mujoco_py
# This is a simple controller written in python for th Lappa model

from math import degrees
import os

STATE = None
RULESTEP = 0
WALKSTEP = 0
TRANSITION_STEP = 0

# SENSOR THRESHOLDS
FIXABLE_THRESHOLD = 0.01
SURFACE_THRESHOLD = 0.5
OBSTACLE_THRESHOLD = 0.15

def deg(scalar):
    return round(degrees(scalar), 2)



def get_sensor_values(module = None):
    global STATE
    sensors = [STATE.sensor('a_h1').data * -1,
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
    global STATE, FIXABLE_THRESHOLD
    if (module == "a"):
        return STATE.sensor("a_rangefinder_down").data < FIXABLE_THRESHOLD
    elif (module == "b"):
        return STATE.sensor("b_rangefinder_down").data < FIXABLE_THRESHOLD
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
        STATE.joint("b_h1").qvel = 0
        STATE.joint("b_h1").qpos = 0
        STATE.joint("b_h1").qacc = 0
        STATE.actuator("a_rotor").ctrl = ctrl
    elif (module == "b"):
        STATE.joint("a_h1").qvel = 0
        STATE.joint("a_h1").qpos = 0
        STATE.joint("a_h1").qacc = 0
        STATE.actuator("b_rotor").ctrl = ctrl
    else:
        print(set_rotate.__name__ + ": Invalid module name")


def is_obstructed(module):
    global STATE, OBSTACLE_THRESHOLD
    if (module == "a"):
        distance = STATE.sensor("a_rangefinder_forward").data
        if (distance < 0):
            return False
        else:
            return distance < OBSTACLE_THRESHOLD
    elif (module == "b"):
        distance = STATE.sensor("b_rangefinder_forward").data
        if (distance < 0):
            return False
        else:
            return distance < OBSTACLE_THRESHOLD
    else:
        print(is_obstructed.__name__ + ": Invalid module name")


def print_actuators():
    global STATE
    a_rotate = STATE.actuator("a_rotor").ctrl
    b_rotate = STATE.actuator("b_rotor").ctrl
    a_thrust = STATE.actuator("a_thrust").ctrl
    b_thrust = STATE.actuator("b_thrust").ctrl
    print("a_rotate: ", a_rotate, "b_rotate: ", b_rotate,
          "a_thrust: ", a_thrust, "b_thrust: ", b_thrust)


def print_sensors():
    sensors = get_sensor_values()
    print("a_h1: ", sensors[0], "b_h1: ", sensors[1],
          "\na_h2: ", sensors[2], "b_h2: ", sensors[3],
          "\na_h3: ", sensors[4], "b_h3: ", sensors[5],
          "\na_rangefinder_forward: ", sensors[6], "b_rangefinder_forward: ", sensors[7],
          "\na_rangefinder_down: ", sensors[8], "b_rangefinder_down: ", sensors[9])


def is_angle(actual_angle, target_angle):
    allowedOffset = 1
    return actual_angle < target_angle + allowedOffset and actual_angle > target_angle - allowedOffset


def set_angle(module, current_angle, target_angle, ctrl = .7):
    if (is_angle(current_angle, target_angle)):
        return
    if (current_angle > target_angle):
        set_rotate(module, -ctrl)
    elif (current_angle < target_angle - 1):
        set_rotate(module, ctrl)
    else:
        set_rotate(module, 0)

def walk_straight(sensors):
    global WALKSTEP
    if (WALKSTEP == 0):
        set_thrust("a", -1)
        set_thrust("b", 0)
        if (is_angle(deg(sensors[0]), 45)):
            set_rotate("a", 0)
            WALKSTEP = 1
        else:
            set_angle("a", sensors[0], 45)
    elif (WALKSTEP == 1):
        set_thrust("a", 0)
        set_thrust("b", -1)
        if (is_angle(deg(sensors[1]), 45)):
            set_rotate("b", 0)
            WALKSTEP = 0
        else:
            set_angle("b", sensors[1], 45)
    else:
        set_thrust("a", -1)
        set_thrust("b", -1)

def prepare_transition(module, sensor):
    counter_module = "a" if module == "b" else "b"
    if (not is_obstructed(module)):
        set_thrust(counter_module, -1)
        set_thrust(module, 0)
        set_rotate(counter_module, .7)
    else:
        if(sensor < 0.14):
            set_thrust(counter_module, -1)
            set_thrust(module, 0)
            set_rotate(counter_module, -.7)
        else:
            set_rotate(counter_module, 0)
            set_thrust(module, -1)

def is_prepared(sensor):
    return sensor < 0.16 and sensor > 0.14

def detect_surface(sensor):
    return sensor <= SURFACE_THRESHOLD

def transition_wall(module, sensor, h1, counter_h1):
    global TRANSITION_STEP
    counter_module = "a" if module == "b" else "b"
    if (TRANSITION_STEP == 0):
        set_thrust(counter_module, -1)
        set_thrust(module, .1)
        if (not detect_surface(sensor)):
            TRANSITION_STEP = 1
            print("Transitioning step 0 complete...")
    if (TRANSITION_STEP == 1):
        set_thrust(module, 0)
        set_thrust(counter_module, -1)
        if (deg(counter_h1) < 90):
            set_thrust(module, .1)
            set_angle(counter_module, deg(counter_h1), 90, .1)
        else:
            set_thrust(module, -1)
            set_rotate(counter_module, 0)
            TRANSITION_STEP = 2
            print("Transitioning step 1 complete...")
    if (TRANSITION_STEP == 2):
        set_rotate(counter_module, 0)
        set_rotate(module, 0)
        set_thrust(module, -1)
        set_thrust(counter_module, 0)
        if (deg(h1) < 90):
            print(deg(h1))
            set_thrust(counter_module, .1)
            set_angle(module, deg(h1), 90, .2)
        else:
            set_thrust(counter_module, -1)
            set_rotate(module, 0)
            TRANSITION_STEP = 3
            print("Transitioning step 2 complete...")
    if (TRANSITION_STEP == 3):
        print("Done.")



def controller(model, data):
    global STATE, RULESTEP, WALKSTEP, TRANSITION_STEP
    STATE = data
    sensors = get_sensor_values()

    if (is_grounded("a") and is_grounded("b") and RULESTEP == 0):
        RULESTEP = 1
    
    if (RULESTEP == 1):
        if (is_obstructed("a") or is_obstructed("b")):
            print("Obstructed", is_obstructed("a"), is_obstructed("b"))
            WALKSTEP = 0
            RULESTEP = 2
        else:
            walk_straight(sensors)

    if (RULESTEP == 2):
        print("Preparing transition")
        set_rotate("a", 0)
        set_rotate("b", 0)
        if (is_prepared(sensors[6])):
            if (is_prepared(sensors[7])):
                set_thrust("b", -1)
                RULESTEP = 3
                print("Finished preparing")
            else:
                prepare_transition("b", sensors[7])
        else:
            prepare_transition("a", sensors[6])

    if (RULESTEP == 3):
        transition_wall("b", sensors[9], sensors[1], sensors[0])
    

    pass
