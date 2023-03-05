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
            API.walk_straight(45, 1)
        else:
            STEP = 2
    elif (STEP == 2):
        on_wall = API.transition_wall("a")
        if (on_wall):
            STEP = 3
    elif (STEP == 3):
        if (API.is_fixed("a") and API.is_fixed("b")):
            STEP = 4
    elif (STEP == 4):
        print("Sucess!")
    else:
        print("Something is wrong...")
    pass