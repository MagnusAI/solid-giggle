import sys
from api_v35 import *

robot = None
actions = []

action = None
state = (False, False, 0, 0, 0, False, False)  # init_state
next_state = None
sensor_delay = 0  # Wait for sensors data to be updated
phase = 1


def get_action(state):
    global phase

    if (phase == 1):
        return phase_one(state)
    elif (phase == 2):
        return phase_two(state)
    elif (phase == 3):
        return phase_three(state)
    elif (phase == 4):
        return phase_four(state)
    elif (phase == 5):
        return phase_five(state)
    elif (phase == 6):
        return phase_six(state)
    else:
        raise Exception("Invalid phase")


def phase_one(state):
    global phase
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    fixed = a_fixed or b_fixed

    if (not fixed):
        return 'lower_a'
    else:
        phase = 2
        return phase_two(state)


def phase_two(state):
    global phase
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    fixed = a_fixed or b_fixed
    angled = arm_angle == 90

    if (not fixed):
        phase = 1
        return phase_one(state)
    else:
        if (not angled):
            return 'lift_b'
        else:
            phase = 3
            return phase_three(state)


def phase_three(state):
    global phase
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    fixed = a_fixed or b_fixed
    angled = arm_angle == 90

    if (not fixed):
        phase = 1
        return phase_one(state)
    elif (not angled):
        phase = 2
        return phase_two(state)
    else:
        if (b_distance > 30):
            return 'rotate_a_backward'
        else:
            phase = 4
            return phase_four(state)


def phase_four(state):
    global phase, action
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state
    fixed = a_fixed or b_fixed
    angled = arm_angle == 90

    if (not fixed):
        phase = 1
        return phase_one(state)
    elif (b_distance > 30):
        phase = 3
        return phase_three(state)
    else:
        if (b_distance == 999):
            return 'rotate_a_forward'
        if (action == 'rotate_a_forward'):
            return 'lower_b'
        if (b_distance > 20):     # Try to change this to values between 0 and 30
            return 'rotate_a_backward'
        if (not b_fixed):
            return 'lower_b'
        else:
            phase = 5
            return phase_five(state)


def phase_five(state):
    global phase
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state

    if (a_fixed and b_fixed):
        return 'lift_a'
    elif (b_fixed and b_leveled):
        phase = 6
        return phase_six(state)
    else:
        phase = 2
        return phase_two(state)


def phase_six(state):
    global phase
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_leveled, b_leveled = state

    fixed = a_fixed or b_fixed
    half_leveled = a_leveled or b_leveled
    leveled = a_leveled and b_leveled

    if (not fixed):
        phase = 4
        return phase_four(state)
    elif (leveled and a_fixed and b_fixed):
        print("Complete.")
        phase = 7
        return 'stop'
    elif (leveled and a_distance > 30):
        return 'lower_a'

    return 'rotate_b_forward'


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.read_state_from_sensors()
    return next_state

def stop():
    sys.exit(0)


def controller(model, data):
    global robot, state, action, actions, sensor_delay, next_state

    if (robot is None):
        robot = LappaApi(data)
        robot.unlock()
        return
    else:
        robot.update_data(data)

    if (sensor_delay == 0):
        if (not robot.is_locked()):
            action = get_action(state)
            print("Action:", action)

        if (action == 'stop'):
            print("Time to complete:", round(data.time, 2), "seconds")
            stop()

        next_state = perform_action(robot, action)
        sensor_delay = 1

        robot.lock()
        if (next_state != state):
            print("New state:", state, "->", next_state)
            robot.unlock()

        if (not robot.is_locked()):
            state = next_state
            actions.append(action)

            # Debug info
            robot.debug_info()
            print("State: ", robot.read_state_from_sensors())
            print("Actions: ", actions)
            print("Phase: ", phase)
            print("___________________________________________________________")
    else:
        sensor_delay -= 1
