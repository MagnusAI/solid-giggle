from api import *

API = None
STEP = 0

def controller(model, data):
    global API, STEP
    if (API is None):
        API = LappaApi(data, .01, .03)
    else:
        API.update_state(data)
    if (STEP == 0):
        if (API.is_fixable("a") and API.is_fixable("b")):
            STEP = 1
    elif (STEP == 1):
        if (API.is_obstructed("a") and API.is_obstructed("b")):
            STEP = 2
        else:
            API.walk_straight(45, 1)
    elif (STEP == 2):
        jumped = API.jump("a")
        if (jumped):
            STEP = 3
    elif (STEP == 3):
        return
    else:
        print("Something is wrong...")
    pass