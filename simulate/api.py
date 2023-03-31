# This API implements the interface_api.py

from interface_api import InterfaceLappaApi


class LappaApi(InterfaceLappaApi):
    def __init__(self, data):
        self.data = data
        self.locket = False

    def update_data(self, data):
        self.data = data

    def is_locked(self):
        return self.locket

    def lock(self):
        self.locket = True

    def unlock(self):
        self.locket = False

    def read_sensors(self, module, sensor_names):
        def get_sensor_value(name):
            values = self.data.sensor(module + "_" + name).data
            if name == "vacuum":
                return [values[2]]
            else:
                return values

        data = map(get_sensor_value, sensor_names)
        return [round(item, 4) for sublist in data for item in sublist]

    def read_actuators(self, module, actuator_names):
        def get_actuator_value(name):
            return self.data.actuator(module + "_" + name).ctrl

        data = map(get_actuator_value, actuator_names)
        return [round(item[0], 4) for item in data]

    def set_actuator(self, module, actuator, ctrl):
        self.data.actuator(module + "_" + actuator).ctrl = ctrl

    def is_fixable(self, module):
        print(self.data.sensor(module + "_vacuum").data)
        pass

    def fix_module(self, module):
        self.data.actuator(module + "_thrust").ctrl = -1
        self.data.actuator(module + "_vacuum").ctrl = 1

    def release_module(self, module, ctrl=.75):
        self.data.actuator(module + "_thrust").ctrl = ctrl
        self.data.actuator(module + "_vacuum").ctrl = 0

    def is_obstructed(self, module):
        #distance = self.data.sensor(module + "_range").data[0]
        #return distance > 0.1
        pass
        
    def rotate_module(self, module, ctrl):
        # Lock the joint of the other module
        counter_module = "a" if module == "b" else "b"
        joint = counter_module + "_h1"
        self.data.joint(joint).qvel = 0
        self.data.joint(joint).qpos = 0
        self.data.joint(joint).qacc = 0

        modifier = 1 if module == "a" else -1
        self.data.actuator(module + "_h1").ctrl = ctrl * modifier

    def stop_rotation(self, module):
        self.data.actuator(module + "_h1").ctrl = 0
        joint = module + "_h1"
        self.data.joint(joint).qvel = 0
        self.data.joint(joint).qpos = 0
        self.data.joint(joint).qacc = 0
