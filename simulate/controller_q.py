import json
from api_v2 import *
import numpy as np

API = None
END = False
STATE = None

# Define the possible actions
ACTIONS = [
    "lift_a",
    "lift_b",
    "lower_a",
    "lower_b",
    "rotate_a_90",
    "rotate_a_270",
    "rotate_b_90",
    "rotate_b_270",
]

# Define the possible states
STATES = [
    (True, True, False, False, False),
    (True, False, False, False, False),
    (True, False, True, False, False),
    (True, False, True, True, False),
    (True, True, True, True, False),
    (False, True, True, True, False),
    (False, True, True, False, False),
    (False, True, False, False, False),
    (False, True, False, True, False),
    (True, True, False, True, False),
    (True, False, False, True, False),
    (True, True, True, False, False),
    (False, False, False, True, False),
    (False, False, True, True, False),
    (False, False, True, False, False),
    (False, False, False, False, False),
    (True, True, False, False, True),
    (False, False, False, True, True),
    (True, False, False, False, True),
    (True, False, True, False, True),
    (True, False, True, True, True),
    (True, True, True, True, True),
    (False, True, True, True, True),
    (False, True, True, False, True),
    (False, True, False, False, True),
    (False, True, False, True, True),
    (True, True, False, True, True),
    (True, False, False, True, True),
    (True, True, True, False, True),
    (False, False, True, True, True),
    (False, False, True, False, True),
    (False, False, False, False, True)
]

# Define the Q-learning table
# If there exist a file at output/q_table.json, load it
# Otherwise, initialize the Q-learning table with zeros
try:
    with open("output/q_table.json", "r") as f:
        Q_TABLE = np.array(json.load(f))
except:
    Q_TABLE = np.zeros((len(STATES), len(ACTIONS)))

# Define the learning parameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPISODES = 10000
EPSILON = 0.1

# Define the reward function


def get_reward(state):
    global AC_TAKEN
    if ((not state[0] and not state[1]) and (AC_TAKEN.__len__() > 0)):
        return -100
    elif not state[4]:
        return -1
    elif state == (True, True, False, False, True):
        return 100
    else:
        return -1

# Choose an action based on the Q-learning table and epsilon-greedy policy


def choose_action(state, epsilon):
    if np.random.uniform() < epsilon:
        action = ACTIONS[np.random.randint(len(ACTIONS))]
    else:
        state_index = STATES.index(state)
        action_index = np.argmax(Q_TABLE[state_index])
        action = ACTIONS[action_index]
    return action

# Update the Q-learning table based on the observed reward and next state


def update_q_table(state, action, reward, next_state):
    state_index = STATES.index(state)
    action_index = ACTIONS.index(action)
    next_state_index = STATES.index(next_state)
    max_q = np.max(Q_TABLE[next_state_index])
    td_target = reward + DISCOUNT_FACTOR * max_q
    td_error = td_target - Q_TABLE[state_index, action_index]
    Q_TABLE[state_index, action_index] += LEARNING_RATE * td_error
    # Store Q-table in a JSON file
    with open("output/q_table.json", "w") as f:
        json.dump(Q_TABLE.tolist(), f)


def get_state():
    global STATE
    a_fixed = API.get_pressure("a") < -100
    b_fixed = API.get_pressure("b") < -100
    lifted = is_lifted()
    rotated = is_rotated()
    levelled = is_levelled()
    state = (a_fixed, b_fixed, lifted, rotated, levelled)

    return state


def lower(module):
    global API
    API.set_thruster(module, -.5)
    pressure = API.get_pressure(module)
    if (pressure < 0):
        API.set_thruster(module, -1)
        API.set_adhesion(module, 1)


def lift(module):
    global API
    API.set_adhesion(module, 0)
    API.set_thruster(module, .5)


def rotate(module, ctrl):
    global API
    counter_module = "b" if module == "a" else "a"
    API.stop_rotation(counter_module)
    API.rotate_module(module, ctrl)


def get_target_angle(module):
    global API
    h1 = API.get_h1(module)

    direction = 90 if h1 < 90 else -90
    target = h1 + direction
    return target % 360 if target > 0 else target + 360


def is_angle(module, target):
    global API
    h1 = round(API.get_h1(module), 1)
    d1 = abs(target - h1)
    d2 = 360 - d1
    diff = min(d1, d2)
    allowed_offset = 5
    return diff < allowed_offset


def rotate_to(module, target, ctrl=.5):
    global API
    if (is_angle(module, target)):
        API.stop_rotation(module)
        return
    else:
        API.set_thruster(module, .1)
        h1 = round(API.get_h1(module), 1)
        diff = abs(h1 - target) % 360
        if diff <= 180:
            ctrl = ctrl if target > h1 else -ctrl
        else:
            ctrl = ctrl if target < h1 else -ctrl
        rotate(module, ctrl)


def is_lifted():
    global API
    a_h2 = API.get_h2("a")
    b_h2 = API.get_h2("b")
    acc = a_h2 + b_h2
    offset = 5
    return acc > (45 - offset)


def is_rotated():
    one_90 = is_angle("a", 90) or is_angle("b", 90)
    both_90 = is_angle("a", 90) and is_angle("b", 90)
    one_270 = is_angle("a", 270) or is_angle("b", 270)
    both_270 = is_angle("a", 270) and is_angle("b", 270)
    rot_90 = one_90 and not both_90
    rot_270 = one_270 and not both_270
    return rot_90 or rot_270


def is_levelled():
    global API
    pos = round(API.get_position()[2], 2)
    return pos > 0.1


def perform_action(action):
    if action == "lift_a":
        lift("a")
    elif action == "lift_b":
        lift("b")
    elif action == "lower_a":
        lower("a")
    elif action == "lower_b":
        lower("b")
    elif action == "rotate_a_90":
        rotate_to("a", 90)
    elif action == "rotate_a_270":
        rotate_to("a", 270)
    elif action == "rotate_b_90":
        rotate_to("b", 90)
    elif action == "rotate_b_270":
        rotate_to("b", 270)
    elif action == "no_op":
        pass
    else:
        print("Invalid action:", action)

    next_state = get_state()

    if (next_state == PREV_STATE):
        return 0, next_state
    else:
        reward = get_reward(next_state)
        return reward, next_state


AC_TAKEN = []
PREV_STATE = None


def controller(model, data):
    global API, END, AC_TAKEN, PREV_STATE

    if (END):
        return

    if (API is None):
        API = LappaApi(data)
        return
    else:
        API.update_data(data)

        state = get_state()

        if (state == PREV_STATE):
            return

        action = choose_action(state, EPSILON)
        reward, next_state = perform_action(action)
        print("action:", action, "reward:", reward, "next_state:", next_state)


        if (len(AC_TAKEN) == 0 or AC_TAKEN[-1] != action):
            AC_TAKEN.append(action)

        if (reward != 0):
            update_q_table(state, action, reward, next_state)
            print("reward:", reward)
            print("Q_TABLE:\n", Q_TABLE)

        if (reward == -100):
            API.reset()
        else:
            pass

        if next_state == (True, True, False, False, True):
            END = True

        PREV_STATE = next_state

        # Print some information for debugging purposes
        # print("state:", state)
        # print("action:", action)
        # print("reward:", reward)
        # print("next_state:", next_state)
        # print("AC_TAKEN:\n", AC_TAKEN)
        # print("Q_TABLE:\n", Q_TABLE)
