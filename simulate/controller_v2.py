from api import *

API = None

def controller(model, data):
    global API
    if (API is None):
        API = LappaApi(data)
    else:
        API.update_state(data)

    if (API.is_fixable("a") and API.is_fixable("b")):
        if (not (API.is_obstructed("a") and API.is_obstructed("b"))):
            API.walk_straight()
        else:
            print("Transitioning...")
            API.transition_wall("a")
    else:
        if (not not (API.is_fixable("a") and API.is_fixable("b"))):
            print("Something is wrong...", API.is_fixable("a"), API.is_fixable("b"))
    
    pass