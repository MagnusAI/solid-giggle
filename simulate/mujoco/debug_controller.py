import os
import sys
from api_v36 import *
from deep_ql_clean import calculate_reward, get_stale_count, perform_action

actions_idx = None
action = None
robot = None
actions = []
rewards = []
action_space = ['lift_a', 'lift_b', 'lower_a', 'lower_b',
                'rotate_a_forward', 'rotate_b_forward', 'rotate_a_backward', 'rotate_b_backward', 'stop_a_rotation', 'stop_b_rotation']
neutral_actions = ['stop_a_rotation', 'stop_b_rotation']
stale_count = 0
stale_limit = 5000
done = False
state = (False, False, 0, 0, 0, False, False)  # init_state
score = 0

todo = ['stop_a_rotation', 'lower_a', 'rotate_a_backward', 'rotate_a_backward', 'rotate_b_backward']




def stop():
    sys.exit(0)


def controller(model, data):
    global robot, state, action, stale_count, stale_limit, actions, score, done, todo, rewards

    if (robot is None):
        robot = LappaApi(data)
        todo.reverse()
        robot.reset()
        state = (False, False, 0, 0, 0, False, False)
        return
    else:
        robot.update_data(data)

    done = len(todo) <= 0 and not robot.is_locked()
    stale_count = get_stale_count()

    if (not done):
        if (not robot.is_locked()):
            action = todo.pop()
            print("Action:", action)

        next_state = perform_action(robot, state, action)

        if (not robot.is_locked() or stale_count == stale_limit):

            reward = calculate_reward(state, next_state, action)

            state = next_state
            actions.append(action)
            rewards.append(reward)
            score += reward

            robot.debug_info()
            print("State: ", state)
            print("Next state: ", next_state)
            print("Actions: ", actions)
            print("Rewards: ", rewards)
            print("Stale count:", stale_count)
            print("Reward:", reward)
            print("Score:", score)
            print("Remaining actions:", todo)
            print(
                "-------------------------------------------------------------------------------------------")
    else:
        #sys.exit(0)
        pass
        

