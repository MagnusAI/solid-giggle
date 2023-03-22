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
        pass

    def fix_module(self, module):
        # print all posible values of the body
        house = self.state.body(module + "_module")
        house.cvel = [0,0,0,0,0,0]
        house.crb = [0,0,0,0,0,0,0,0,0,0]
        house.xpos= [ 0, 0, 0]
        print(house)

        #print(body)
        # set thrust to -1 and fix the position of the module making it static
        self.state.actuator(module + "_thrust").ctrl = -1
        
    def print_test(self):
        part = self.state.body("a_chamber")
        print(part)

    def release_module(self, module, ctrl=0):
        self.state.actuator(module + "_thrust").ctrl = ctrl
        self[module + "pos"] = None

    def is_obstructed(self, module):
        pass

    def rotate_module(self, module, degrees, ctrl):
        pass

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
