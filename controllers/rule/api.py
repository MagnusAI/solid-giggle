import math
import random

AXIS = 2  # 0 = x, 1 = y, 2 = z

class LappaApi():
    def __init__(self, data):
        self.data = data
        self.locked = False

    def is_locked(self):
        return self.locked

    def lock(self):
        self.locked = True

    def unlock(self):
        self.locked = False

    def debug_info(self):
        print("State: ", self.read_state_from_sensors())
        print("H1: ", round(self.get_h1("a"), 1),
              " , ", round(self.get_h1("b"), 1))
        print("H2: ", round(self.get_h2("a"), 2),
              " , ", round(self.get_h2("b"), 2))
        print("Pressure: ", round(self.get_pressure("a"), 1),
              " , ", round(self.get_pressure("b"), 1))
        print("Thrust: ", self.get_thruster(
            "a"), " , ", self.get_thruster("b"))
        print("Adhesion: ", self.get_adhesion(
            "a"), " , ", self.get_adhesion("b"))
        print("Rotation: ", self.get_h1_actuator(
            "a"), " , ", self.get_h1_actuator("b"))
        print("Range: ", round(self.get_range("a"), 2),
              " , ", round(self.get_range("b"), 2))
        print("Position", round(self.get_position("a")[AXIS], 2), round(
            self.get_position("b")[AXIS], 2))

    def update_data(self, data):
        self.data = data
    
    def get_data(self):
        return self.data

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

    def get_range(self, module):
        data = self.data.sensor(module + "_rangefinder").data[0]
        noise = random.gauss(0, 0.05)  # Mean = 0, std_dev = 0.05 (change according to your needs)
        data += noise  # Add noise to the sensor value
        cm = data * 100
        if (self.get_pressure(module) > 2):
            return cm - 5  # The pressure somehow messes up the range
        return cm

    def get_h1(self, module):
        scalar = self.data.sensor(module + "_h1").data[0]
        noise = random.gauss(0, 0.05)  # Mean = 0, std_dev = 0.05 (change according to your needs)
        scalar += noise  # Add noise to the sensor value
        degrees = math.degrees(scalar) % 360
        if (degrees < 0):
            degrees += 360
        return degrees

    def get_h2(self, module):
        sensor_value = self.data.sensor(module + "_h2").data[0]
        noise = random.gauss(0, 0.05)  # Mean = 0, std_dev = 0.05 (change according to your needs)
        sensor_value += noise  # Add noise to the sensor value

        # Convert the sensor value to degrees using a linear mapping
        degrees = (sensor_value * 45) / 0.785

        # Round the result to two decimal places
        degrees = round(degrees, 2)

        if (module == "b"):
            degrees = -degrees
        return degrees

    def get_pressure(self, module):
        data = self.data.sensor(module + "_vacuum").data[2]
        noise = random.gauss(0, 0.05)  # Mean = 0, std_dev = 0.05 (change according to your needs)
        data += noise  # Add noise to the sensor value
        return data

    def stop_rotation(self, module):
        self.data.actuator(module + "_h1").ctrl = 0
        self.data.joint(module + "_h1").qvel = 0
        self.data.joint(module + "_h1").qacc = 0

    def rotate_module(self, module, ctrl):
        self.data.actuator(module + "_h1").ctrl = ctrl

    def reset_h1(self, module):
        self.data.actuator(module + "_h1").ctrl = 0
        self.data.joint(module + "_h1").qvel = 0
        self.data.joint(module + "_h1").qpos = 0
        self.data.joint(module + "_h1").qacc = 0

    def get_position(self, module):
        pos = self.data.sensor(module + "_position").data
        return pos

    def lift(self, module):
        self.set_adhesion(module, 0)
        self.set_thruster(module, .35)
        self.reset_h1(module)

    def lower(self, module):
        self.set_thruster(module, -1)

    def rotate(self, module, ctrl):
        counter_module = "b" if module == "a" else "a"
        self.stop_rotation(counter_module)
        self.rotate_module(module, ctrl)

    def rotate_forward(self, module):
        self.rotate(module, .2)

    def rotate_backward(self, module):
        self.rotate(module, -.2)

    def update_pressure(self):
        a_thrust = self.get_thruster("a")
        b_thrust = self.get_thruster("b")
        a_pressure = self.get_pressure("a")
        b_pressure = self.get_pressure("b")
        if (a_pressure > 2 and a_thrust < 0):
            self.set_adhesion("a", .55)
        else:
            self.set_adhesion("a", 0)
        if (b_pressure > 2 and b_thrust < 0):
            self.set_adhesion("b", .55)
        else:
            self.set_adhesion("b", 0)

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
        else:
            print("Unknown action: " + action)
        self.update_pressure()

    def get_arm_angle(self):
        a_h2 = self.get_h2("a")
        b_h2 = self.get_h2("b")
        acc = a_h2 + b_h2

        if acc > 80:
            return 90
        elif acc > 65:
            return 75
        elif acc > 50:
            return 60
        elif acc > 35:
            return 45
        elif acc > 20:
            return 30
        elif acc > 10:
            return 15
        else:
            return 0

    def get_distance(self, module):
        range = self.get_range(module)

        if (range < -99):
            return 999
        elif (range <= 0.99):
            return 0
        elif (range <= 2):
            return 2
        elif (range <= 4):
            return 4
        elif (range <= 6):
            return 6
        elif (range <= 8):
            return 8
        elif (range <= 10):
            return 10
        elif (range <= 12):
            return 12
        elif (range <= 14):
            return 14
        elif (range <= 16):
            return 16
        elif (range <= 18):
            return 18
        elif (range <= 20):
            return 20
        else:
            return 30

    def read_state_from_sensors(self):
        global AXIS
        a_fixed = self.get_pressure("a") > 45
        b_fixed = self.get_pressure("b") > 45
        arm_angle = self.get_arm_angle()
        a_range = self.get_distance("a")
        b_range = self.get_distance("b")
        a_leveled = round(self.get_position("a")[AXIS], 2) > 0.15
        b_leveled = round(self.get_position("b")[AXIS], 2) > 0.15
        return (a_fixed, b_fixed, arm_angle, a_range, b_range, a_leveled, b_leveled)
