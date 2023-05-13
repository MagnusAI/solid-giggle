import sys
from api import *

robot = None
actions = []

action = None
state = (False, False, 0, 0, 0, False, False)  # init_state
next_state = None
phase = 1
start_phase_time = 0


def get_action(phase):
    if (phase == 1):
        return 'lower_a'
    elif (phase == 2):
        return 'lift_b'
    elif (phase == 3):
        return 'rotate_a_backward'
    elif (phase == 4):
        return 'lower_b'
    elif (phase == 5):
        return 'lift_a'
    elif (phase == 6):
        return 'rotate_b_forward'
    elif (phase == 7):
        return 'lower_a'
    else:
        return 'stop'

def get_duration(phase):
    if (phase == 1):
        return 1
    elif (phase == 2):
        return 1
    elif (phase == 3):
        return 1.5
    elif (phase == 4):
        return 1
    elif (phase == 5):
        return .1
    elif (phase == 6):
        return .5
    elif (phase == 7):
        return 1
    else:
        return 1


def perform_action(robot, action):
    robot.perform_action(action)


def controller(model, data):
    global robot, state, action, actions, sensor_delay, next_state, phase, start_phase_time

    if (robot is None):
        robot = LappaApi(data)
        robot.unlock()
        start_phase_time = data.time
        return
    else:
        robot.update_data(data)

        if (not robot.is_locked()):
            duration = get_duration(phase)
            if data.time - start_phase_time >= duration:
                robot.unlock()
                phase += 1
                start_phase_time = data.time
                # Debug info
                print("--------------------------------- Phase", phase, "---------------------------------")
                robot.debug_info()

            action = get_action(phase)

        if (action == 'stop'):
            print("Time to complete:", round(data.time, 2), "seconds")
            sys.exit(0)

        next_state = perform_action(robot, action)
