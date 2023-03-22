# This API implements the interface_api.py 

import numpy as np
import sys

MAX_INT = sys.maxsize

from interface_api import InterfaceLappaApi

class LappaApi(InterfaceLappaApi):
    def __init__(self, state):
        self.state = state
        self.apos = None
        self.bpos = None

    def update_state(self, state):
        self.state = state

    def read_sensors(self, module, sensor_names):
        def get_sensor_value(name):
            return self.state.sensor(module + "_" + name).data
        
        data = map(get_sensor_value, sensor_names)
        return [round(item, 4) for sublist in data for item in sublist]

    def read_actuators(self, module, actuator_names):
        def get_actuator_value(name):
            return self.state.actuator(module + "_" + name).ctrl
        
        data = map(get_actuator_value, actuator_names)
        return [round(item[0], 4) for item in data]

    def set_actuator(self, module, actuator, ctrl):
        self.state.actuator(module + "_" + actuator).ctrl = ctrl

    def is_fixable(self, module):
        print(self.state.sensor(module + "_vacuum").data)
        pass

    def fix_module(self, module):
        self.state.actuator(module + "_thrust").ctrl = -1
        self.state.actuator(module + "_vacuum").ctrl = 1

    def release_module(self, module, ctrl=0):
        self.state.actuator(module + "_thrust").ctrl = 0.5
        self.state.actuator(module + "_vacuum").ctrl = 0

    def is_obstructed(self, module):
        distance = self.state.sensor(module + "_range").data[0]
        return distance > 0.1

    def rotate_module(self, module, degrees, ctrl):
        # fix module
        self.fix_module(module)
        # release other module
        self.release_module("a" if module == "b" else "b")
        # rotate module
        self.state.actuator(module + "_h1").ctrl = ctrl

    def stop(self):
        pass
    
    def walk(self, ctrl, angle, release_ctrl=0):
        pass

    def jump(self, ctrl, release_ctrl=0):
        pass

    def climb(self, ctrl, release_ctrl=0):
        pass

    def descend(self, ctrl, release_ctrl=0):
        pass
