import os
import sys
from api_v36 import *
from deep_ql_v5 import calculate_reward, get_stale_count, perform_action

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

todo = ['lower_a', 'lift_b', 'rotate_a_backward', 'rotate_a_backward', 'lower_b']
episodes = 5

def print_statistics():
    global robot, actions, rewards, stale_count, score, todo
    robot.debug_info()
    print("State: ", state)
    print("Actions: ", actions)
    print("Rewards: ", rewards)
    print("Stale count:", stale_count)
    print("Score:", score)
    print("Remaining actions:", todo)
    print(
        "-------------------------------------------------------------------------------------------")

def reset():
    global episodes, robot, state, stale_count, score, actions, rewards, todo
    print_statistics()
    episodes -= 1
    if (episodes > 0):
        todo = ['lower_a', 'lift_b', 'rotate_a_backward', 'rotate_a_backward', 'lower_b']
    else:
        sys.exit()
    score = 0
    actions.clear()
    rewards.clear()
    stale_count = 0
    state = (False, False, 0, 0, 0, False, False)  # init_state
    
def get_done():
    global done
    return done

def set_done(value):
    global done
    done = value

def controller(model, data):
    global robot, state, action, stale_count, stale_limit, actions, score, done, todo, rewards

    if (robot is None):
        robot = LappaApi(data)
        todo.reverse()
        state = (False, False, 0, 0, 0, False, False)
        return
    else:
        robot.update_data(data)

    done = len(todo) <= 0 and not robot.is_locked()
    stale_count = get_stale_count()

    if (not done):
        if (not robot.is_locked()):
            action = todo.pop()

        next_state = perform_action(robot, action)
        if (next_state != state):
            stale_count = 0
            robot.unlock()

        if (not robot.is_locked() or stale_count == stale_limit):

            reward = calculate_reward(state, next_state, action)

            state = next_state
            actions.append(action)
            rewards.append(reward)
            score += reward
    else:
        pass
        

