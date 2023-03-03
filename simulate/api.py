from math import degrees

def deg(scalar):
    return round(degrees(scalar), 2)
    
class CoreApi:
    def __init__(self, state):
        self.state = state

    def update_state(self, state):
        self.state = state

    def get_sensor(self, sensor_name):
        data = self.state.sensor(sensor_name).data
        if (sensor_name == "a_h1"):
            return [data[0] * -1]
        else:        
            return data
    
    def read_sensors(self, sensor_names):
        sensors = map(self.get_sensor, sensor_names)
        return [round(item, 4) for sublist in sensors for item in sublist]

    def set_actuator(self, actuator, ctrl):
        self.state.actuator(actuator).ctrl = ctrl
    
    def read_actuator(self, actuator):
        return self.state.actuator(actuator).ctrl


class LappaApi:
    def __init__(self, state, FIXABLE_THRESHOLD = 0.01, OBSTACLE_THRESHOLD = 0.15):
        self.state = state
        self.core = CoreApi(state)
        self.FIXABLE_THRESHOLD = FIXABLE_THRESHOLD
        self.OBSTACLE_THRESHOLD = OBSTACLE_THRESHOLD
        self.WALKSTEP = 0
        self.TRANSITION_WALLSTEP = 0
    
    def update_state(self, state):
        self.state = state

    def get_sensor_values(self, module):
        sensors = ["h1", "h2", "h3", "rangefinder_forward", "rangefinder_down"]
        sensor_names = [module + "_" + sensor for sensor in sensors]
        return self.core.read_sensors(sensor_names)

    def is_fixable(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_down"])[0]
        return sensor_data <= self.FIXABLE_THRESHOLD

    def fix_module(self, module):
        thruster = module + "_thrust"
        self.core.set_actuator(thruster, -1)

    def is_fixed(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_down"])[0]
        actuator_data = self.core.read_actuator(module + "_thrust")
        return sensor_data >= self.FIXABLE_THRESHOLD and actuator_data == -1

    def release_module(self, module):
        thruster = module + "_thrust"
        self.core.set_actuator(thruster, 0)

    def rotate_module(self, module, ctrl):
        rotor = module + "_rotor"
        self.core.set_actuator(rotor, ctrl)
    
    def set_angle(self, module, target_angle, ctrl = .5):
        sensor_data = self.core.read_sensors([module + "_h1"])[0]
        current_angle = deg(sensor_data)
        counter_module = "b" if module == "a" else "a"
        self.lock_joint(counter_module)
        if (current_angle < target_angle):
            self.rotate_module(module, ctrl)
        elif (current_angle > target_angle + 1):
            self.rotate_module(module, -ctrl)
        else:
            self.rotate_module(module, 0)

    def get_angle(self, module):
        sensor_data = self.core.read_sensors([module + "_h1"])[0]
        return deg(sensor_data)
    
    def get_distance(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_forward"])[0]
        return sensor_data
    
    def lock_joint(self, module):
        joint = module + "_h1"
        self.state.joint(joint).qvel = 0
        self.state.joint(joint).qpos = 0
        self.state.joint(joint).qacc = 0

    def walk_straight(self, angle = 45, ctrl = .7):
        if (self.WALKSTEP == 0):
            if (round(self.get_angle("a"), 0) == angle):
                self.rotate_module("a", 0)
                self.WALKSTEP = 1
            else:
                self.fix_module("a")
                self.release_module("b")
                self.set_angle("a", angle, ctrl)
        elif (self.WALKSTEP == 1):
            if (round(self.get_angle("b"), 0) == angle):
                self.rotate_module("b", 0)
                self.WALKSTEP = 0
            else:
                self.fix_module("b")
                self.release_module("a")
                self.set_angle("b", angle, ctrl)

    def is_obstructed(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_forward"])[0]
        return sensor_data <= self.OBSTACLE_THRESHOLD
    
    def transition_wall(self, module, ctrl = .7):
        if (self.TRANSITION_WALLSTEP == 0):
            if (self.is_obstructed("a")):
                self.fix_module("b")
                self.release_module("a")
                self.rotate_module("b", -.7)
            else:
                self.rotate_module("b", 0)
                self.TRANSITION_WALLSTEP = 1
                print("Done with step 0")
        elif (self.TRANSITION_WALLSTEP == 1):
            if (self.is_obstructed("b")):
                self.fix_module("a")
                self.release_module("b")
                self.rotate_module("a", -.7)
            else:
                self.rotate_module("a", 0)
                print("Done with step 1")


    





    
    

    