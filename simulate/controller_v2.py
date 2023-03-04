from api import *

API = None
STEP = 0

def controller(model, data):
    global API, STEP
    if (API is None):
        API = LappaApi(data)
    else:
        API.update_state(data)

    if (STEP == 0):
        if (API.is_fixable("a") and API.is_fixable("b")):
            STEP = 1
    elif (STEP == 1):
        if (not (API.is_obstructed("a") and API.is_obstructed("b"))):
            API.walk_straight()
        else:
            STEP = 2
    elif (STEP == 2):
        API.transition_wall("a")
    else:
        print("Something is wrong...")

    pass