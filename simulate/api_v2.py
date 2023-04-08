import math
from interface_api_v2 import InterfaceLappaApi


class LappaApi(InterfaceLappaApi):
    def __init__(self, data):
        self.data = data

    def update_data(self, data):
        self.data = data

    def set_thruster(self, module, ctrl):
        self.data.actuator(module + "_thrust").ctrl = ctrl

    def get_thruster(self, module):
        return self.data.actuator(module + "_thrust").ctrl[0]

    def set_adhesion(self, module, value):
        self.data.actuator(module + "_vacuum").ctrl = value

    def get_adhesion(self, module):
        return self.data.actuator(module + "_vacuum").ctrl[0]
    
    def get_h1_actuator(self, module):
        return self.data.actuator(module + "_h1").ctrl[0]

    def get_h1(self, module):
        scalar = self.data.sensor(module + "_h1").data[0]
        degrees = math.degrees(scalar) % 360
        if (degrees < 0):
            degrees += 360
        return degrees

    def get_h2(self, module):
        scalar = self.data.sensor(module + "_h2").data[0]
        degrees = math.degrees(scalar)
        if (module == "b"):
            degrees = -degrees
        return degrees

    def get_pressure(self, module):
        return self.data.sensor(module + "_vacuum").data[2]

    def get_touch(self, module):
        return self.data.sensor(module + "_touch").data[0]

    def stop_rotation(self, module):
        self.data.actuator(module + "_h1").ctrl = 0
        self.data.joint(module + "_h1").qvel = 0
        self.data.joint(module + "_h1").qacc = 0

    def rotate_module(self, module, ctrl):
        self.data.actuator(module + "_h1").ctrl = ctrl

    def reset_module(self, module):
        self.data.actuator(module + "_h1").ctrl = 0
        self.data.joint(module + "_h1").qvel = 0
        self.data.joint(module + "_h1").qpos = 0
        self.data.joint(module + "_h1").qacc = 0

    def get_position(self):
        return self.data.sensor("position").data
