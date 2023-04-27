import os
import sys
from api_v3 import *
from deep_ql_v3 import get_reward, perform_action

robot = None
actions = []
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward']
actions_idx = None
stale_count = 0
stale_limit = 5000
state = (False, False, 0, 0, 0, False, False)  # init_state
sensor_delay = 0  # Wait for sensors data to be updated

# todo = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6,
#        6, 6, 6, 3, 3, 3, 3, 3, 3, 0, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4]
#todo = [2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 6, 6, 6, 6, 6, 6, 6 ,6 ,6 ,6 ,3, 3,3,3,3,3,0, 7 ]
todo = [5,3,5]
score = 0
done = False


def stop():
    sys.exit(0)


def controller(model, data):
    global robot, state, action_idx, stale_count, stale_limit, actions, todo, score, sensor_delay, done

    if (robot is None):
        robot = LappaApi(data)
        todo.reverse()
        return
    else:
        robot.update_data(data)

    if (sensor_delay == 0):
        if (not done):
            if (not robot.is_locked()):
                action_idx = todo.pop()

            action = action_space[action_idx]
            next_state = perform_action(robot, action)
            stale_count += 1
            sensor_delay = 1

            robot.lock()
            if (next_state != state):
                print("New state:", state, "->", next_state)
                robot.unlock()
                stale_count = 0
                if (len(todo) == 0):
                    done = True
                else:
                    done = False

            if (not robot.is_locked() or stale_count == stale_limit):

                reward = get_reward(state, next_state)

                if (stale_count == stale_limit):
                    reward = -100

                state = next_state

                actions.append(action)

                score += reward
                # Debug info

                robot.debug_info()
                print("State: ", robot.read_state_from_sensors())
                print("Actions: ", actions.pop())
                print("Stale count:", stale_count)
                print("Reward:", reward)
                print("Score:", score)
                print("Remaining actions:", todo)
                print("Done:", done)
                if (reward < 0):
                    print(
                        "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                print(
                    "-------------------------------------------------------------------------------------------")

                if (reward == 100 or reward == -100):
                    print("Stopped due to reward", reward,
                          "Stale count:", stale_count)
                    stop()
        else:
            # sys.exit(0)
            pass
    else:
        sensor_delay -= 1
