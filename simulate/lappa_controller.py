from lappa_api import *

API = None

ACTUATORS = ["thrust", "h1"]


def controller(model, data):
    global API
    if (API is None):
        API = LappaApi(data)
    else:
        API.update_state(data)

    # if all values in reads accumulated are less than 0.5 then the module is fixable
    


    pass
