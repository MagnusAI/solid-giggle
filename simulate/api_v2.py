import math
from interface_api_v2 import InterfaceLappaApi

AXIS = 2  # 0 = x, 1 = y, 2 = z


class LappaApi(InterfaceLappaApi):
    def __init__(self, data):
        self.data = data
        self.locked = False

    def locked(self):
        return self.locked

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def debug_info(self):
        print("---------------------------------------------------------")
        print("State: ", self.read_state_from_sensors())
        print("H1: ", round(self.get_h1("a"), 1),
              " , ", round(self.get_h1("b"), 1))
        print("H2: ", round(self.get_h2("a"), 1),
              " , ", round(self.get_h2("b"), 1))
        print("Pressure: ", round(self.get_pressure("a"), 1),
              " , ", round(self.get_pressure("b"), 1))
        print("Thrust: ", self.get_thruster(
            "a"), " , ", self.get_thruster("b"))
        print("Adhesion: ", self.get_adhesion(
            "a"), " , ", self.get_adhesion("b"))
        print("Rotation: ", self.get_h1_actuator(
            "a"), " , ", self.get_h1_actuator("b"))

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

    def get_position(self, module):
        pos = self.data.sensor(module + "_position").data
        return pos

    def reset(self):
        self.stop_rotation("a")
        self.stop_rotation("b")
        self.set_thruster("a", 0)
        self.set_thruster("b", 0)
        self.reset_module("a")
        self.reset_module("b")
        self.data.qpos[:] = 0
        self.data.qpos[2] = 0.15
        self.data.qvel[:] = 0
        self.data.qacc[:] = 0
        pass

    def lift(self, module):
        self.set_adhesion(module, 0)
        self.set_thruster(module, .2)
        self.reset_module(module)

    def lower(self, module):
        self.set_thruster(module, -1)
        pressure = self.get_pressure(module)
        if (pressure < -2):
            self.set_adhesion(module, .55)

    def is_angle(self, module, target):
        h1 = round(self.get_h1(module), 1)
        d1 = abs(target - h1)
        d2 = 360 - d1
        diff = min(d1, d2)
        allowed_offset = 5
        return diff < allowed_offset

    def rotate(self, module, ctrl):
        counter_module = "b" if module == "a" else "a"
        self.stop_rotation(counter_module)
        self.rotate_module(module, ctrl)

    def rotate_to(self, module, target, ctrl=.5):
        if (self.is_angle(module, target)):
            self.stop_rotation(module)
            return
        else:
            h1 = round(self.get_h1(module), 1)
            diff = abs(h1 - target) % 360
            if diff <= 180:
                ctrl = ctrl if target > h1 else -ctrl
            else:
                ctrl = ctrl if target < h1 else -ctrl
            self.rotate(module, ctrl)

    def rotate_forward(self, module):
        self.rotate(module, .2)

    def rotate_backward(self, module):
        self.rotate(module, -.2)

    def perform_action(self, action):
        if action == "lift_a":
            self.lift("a")
        elif action == "lift_b":
            self.lift("b")
        elif action == "lower_a":
            self.lower("a")
        elif action == "lower_b":
            self.lower("b")
        elif action == "rotate_a_forward":
            self.rotate_forward("a")
        elif action == "rotate_a_backward":
            self.rotate_backward("a")
        elif action == "rotate_b_forward":
            self.rotate_forward("b")
        elif action == "rotate_b_backward":
            self.rotate_backward("b")
        elif action == "stop_a_rotation":
            self.stop_rotation("a")
        elif action == "stop_b_rotation":
            self.stop_rotation("b")
        elif action == "stop_rotation":
            self.stop_rotation("a")
            self.stop_rotation("b")
        elif action == "stop":
            self.stop_rotation("a")
            self.stop_rotation("b")
            self.set_thruster("a", 0)
            self.set_thruster("b", 0)
        elif action == "reset":
            self.reset()
        else:
            print("Unknown action: " + action)

    def is_lifted(self):
        a_h2 = self.get_h2("a")
        b_h2 = self.get_h2("b")
        acc = a_h2 + b_h2
        offset = 5
        return acc > (45 - offset)

    def is_rotated(self):
        a_h1 = self.get_h1("a")
        b_h1 = self.get_h1("b")
        diff = abs(a_h1 - b_h1)
        if diff > 180:
            diff = 360 - diff
        offset = 10
        return diff > (90 - offset)

    def read_state_from_sensors(self):
        global AXIS
        a_fixed = self.get_pressure("a") < -45
        b_fixed = self.get_pressure("b") < -45

        lifted = self.is_lifted()
        rotated = self.is_rotated()
        a_leveled = round(self.get_position("a")[AXIS], 2) > 0.15
        b_leveled = round(self.get_position("b")[AXIS], 2) > 0.15
        return (a_fixed, b_fixed, lifted, rotated, a_leveled, b_leveled)
