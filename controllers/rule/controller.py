import sys
from api import *

robot = None
actions = []

action = None
state = (False, False, 0, 0, 0, False, False)  # init_state
next_state = None
sensor_delay = 0  # Wait for sensors data to be updated
phase = 1

recursion_depth = 0
max_recursion_depth = 3000  # choose a sensible value


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
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state
    fixed = a_fixed or b_fixed

    if (not fixed):
        return 'lower_a'
    else:
        phase = 2
        return phase_two(state)


def phase_two(state):
    global phase, recursion_depth
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state
    fixed = a_fixed or b_fixed
    angled = arm_angle == 90

    if (not fixed):
        recursion_depth += 1
        phase = 1
        return phase_one(state)
    else:
        if (not angled):
            return 'lift_b'
        else:
            phase = 3
            return phase_three(state)


def phase_three(state):
    global phase, recursion_depth
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state
    fixed = a_fixed or b_fixed
    angled = arm_angle == 90

    if (not fixed):
        recursion_depth += 1
        phase = 1
        return phase_one(state)
    elif (not angled):
        recursion_depth += 1
        phase = 2
        return phase_two(state)
    else:
        if (b_distance > 30):
            return 'rotate_a_backward'
        else:
            phase = 4
            return phase_four(state)


def phase_four(state):
    global phase, action, recursion_depth
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state
    fixed = a_fixed or b_fixed
    angled = arm_angle == 90

    if (not fixed):
        recursion_depth += 1
        phase = 1
        return phase_one(state)
    elif (b_distance > 30):
        recursion_depth += 1
        phase = 3
        return phase_three(state)
    else:
        if (b_distance > 18):     # Try to change this to values between 0 and 30
            return 'rotate_a_backward'
        if (not b_fixed):
            return 'lower_b'
        else:
            phase = 5
            return phase_five(state)


def phase_five(state):
    global phase, recursion_depth
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state

    if (a_fixed and b_fixed):
        return 'lift_a'
    elif (b_fixed and b_levelled):
        phase = 6
        return phase_six(state)
    else:
        recursion_depth += 1
        phase = 2
        return phase_two(state)


def phase_six(state):
    global phase, recursion_depth
    a_fixed, b_fixed, arm_angle, a_distance, b_distance, a_levelled, b_levelled = state

    fixed = a_fixed or b_fixed
    half_levelled = a_levelled or b_levelled
    levelled = a_levelled and b_levelled

    if (not fixed):
        recursion_depth += 1
        phase = 4
        return phase_four(state)
    elif (levelled and a_fixed and b_fixed):
        print("Complete.")
        phase = 7
        return 'stop'
    elif (levelled):
        return 'lower_a'

    return 'rotate_b_forward'


def perform_action(robot, action):
    robot.perform_action(action)
    next_state = robot.get_state()
    return next_state

def stop(data):
    # Log for Testing
    with open("output/results.txt", "a") as f:
        time = round(data.time,2)
        state = robot.get_state()
        success = state[0] and state[1] and state[5] and state[6]
        result = {"Final state": state, "time": time, "success": "True" if success else "False"}
        f.write(str(result) + "\n")
    sys.exit(0)


def controller(model, data):
    global robot, state, action, actions, sensor_delay, next_state

    if recursion_depth > max_recursion_depth:
        print("Recursion depth exceeded")
        stop(data)

    if (robot is None):
        robot = LappaApi(data)
        robot.unlock()
        return
    else:
        robot.update_data(data)

    if (not robot.is_locked()):
        action = get_action(state)
        #print("Action:", action)

    if (action == 'stop' or data.time > 45 or (state[0] and state[1] and state[5] and state[6])):
        print("Time:", round(data.time, 2), "seconds")
        stop(data)

    next_state = perform_action(robot, action)

    robot.lock()
    if (next_state != state):
        #print("New state:", state, "->", next_state)
        robot.unlock()

    if (not robot.is_locked()):
        state = next_state
        actions.append(action)

        # # Debug info
        # robot.debug_info()
        # print("State: ", robot.get_state())
        # print("Actions: ", actions)
        # print("Phase: ", phase)
        # print("___________________________________________________________")

