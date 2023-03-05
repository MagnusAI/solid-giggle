from math import degrees

def deg(scalar):
    return round(degrees(scalar), 2)

def cm(scalar):
    return round(scalar * 100, 2)
    
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
        
    def get_actuator(self, actuator):
        return self.state.actuator(actuator).ctrl

    def set_actuator(self, actuator, ctrl):
        self.state.actuator(actuator).ctrl = ctrl

    def read_sensors(self, sensor_names):
        sensors = map(self.get_sensor, sensor_names)
        return [round(item, 4) for sublist in sensors for item in sublist]

    def read_actuators(self, actuator_names):
        actuators = map(self.get_actuator, actuator_names)
        return [round(item[0], 4) for item in actuators]

    

class LappaApi:
    def __init__(self, state, FIXABLE_THRESHOLD = 0.01, OBSTACLE_THRESHOLD = 0.15):
        self.state = state
        self.core = CoreApi(state)
        self.FIXABLE_THRESHOLD = FIXABLE_THRESHOLD
        self.OBSTACLE_THRESHOLD = OBSTACLE_THRESHOLD
        self.WALKSTEP = 0
        self.DISTANCE_STEP = 0
        self.TRANSITION_WALLSTEP = 0
        self.CLIMBSTEP = 0
    
    def update_state(self, state):
        self.state = state

    def get_sensor_values(self, module):
        sensors = ["h1", "h2", "h3", "rangefinder_forward", "rangefinder_down"]
        sensor_names = [module + "_" + sensor for sensor in sensors]
        return self.core.read_sensors(sensor_names)
    
    def get_actuator_values(self, module):
        actuators = ["thrust", "rotor"]
        actuator_names = [module + "_" + actuator for actuator in actuators]
        return self.core.read_actuators(actuator_names)

    def is_fixable(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_down"])[0]
        return sensor_data <= self.FIXABLE_THRESHOLD

    def fix_module(self, module):
        thruster = module + "_thrust"
        self.core.set_actuator(thruster, -1)

    def is_fixed(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_down"])[0]
        actuator_data = self.core.read_actuators([module + "_thrust"])[0]
        return sensor_data <= self.FIXABLE_THRESHOLD and actuator_data == -1

    def release_module(self, module):
        thruster = module + "_thrust"
        self.core.set_actuator(thruster, 0)

    def rotate_module(self, module, ctrl):
        rotor = module + "_rotor"
        self.core.set_actuator(rotor, ctrl)

    def set_propeller(self, module, ctrl):
        thruster = module + "_thrust"
        self.core.set_actuator(thruster, ctrl)
    
    def set_h1_angle(self, module, target_angle, ctrl = .5):
        sensor_data = self.core.read_sensors([module + "_h1"])[0]
        current_angle = deg(sensor_data)
        counter_module = "b" if module == "a" else "a"
        self.lock_joint(counter_module)
        if (current_angle < target_angle):
            self.rotate_module(module, ctrl)
            return False
        elif (current_angle > target_angle + 1):
            self.rotate_module(module, -ctrl)
            return False
        else:
            self.rotate_module(module, 0)
            return True
    
    def set_height(self, module, target_height, ctrl = .5):
        current_height = cm(self.core.read_sensors([module + "_rangefinder_down"])[0])
        if (current_height < target_height):
            self.core.set_actuator(module + "_thrust", ctrl)
            return False
        elif (current_height > target_height + 0.01):
            self.core.set_actuator(module + "_thrust", -.1)
            return False
        else:
            self.core.set_actuator(module + "_thrust", 0)
            return True

    def get_angle(self, module, sensor = "h1"):
        sensor_data = self.core.read_sensors([module + "_" + sensor])[0]
        return deg(sensor_data)
    
    def get_distance(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_forward"])[0]
        return sensor_data
    
    def lock_joint(self, module):
        joint = module + "_h1"
        self.state.joint(joint).qvel = 0
        self.state.joint(joint).qpos = 0
        self.state.joint(joint).qacc = 0

    def walk_straight(self, angle = 45, ctrl = 1, zctrl = 0):
        if (self.WALKSTEP == 0):
            if (round(self.get_angle("a"), 0) == angle):
                self.rotate_module("a", 0)
                self.WALKSTEP = 1
            else:
                self.fix_module("a")
                if (zctrl != 0):
                    self.set_propeller("b", zctrl)
                else:
                    self.release_module("b")
                self.set_h1_angle("a", angle, ctrl)
        elif (self.WALKSTEP == 1):
            if (round(self.get_angle("b"), 0) == angle):
                self.rotate_module("b", 0)
                self.WALKSTEP = 0
            else:
                self.fix_module("b")
                if (zctrl != 0):
                    self.set_propeller("a", zctrl)
                else:
                    self.release_module("a")
                self.set_h1_angle("b", angle, ctrl)

    def is_obstructed(self, module):
        sensor_data = self.core.read_sensors([module + "_rangefinder_forward"])[0]
        return sensor_data < self.OBSTACLE_THRESHOLD and sensor_data >= 0
    
    def set_distance(self, module, target_distance, ctrl = -1):
        counter_module = "b" if module == "a" else "a"
        distance = self.get_distance(module)
        if (distance < target_distance):
            self.fix_module(counter_module)
            self.release_module(module)
            self.rotate_module(module, 0)
            self.rotate_module(counter_module, ctrl)
            return False
        else:
            self.rotate_module(counter_module, 0)
            self.fix_module(module)
            self.fix_module(counter_module)
            return True
        
    def set_h2_angle(self, module, target_angle, ctrl = .5):
        sensor_data = self.core.read_sensors([module + "_h2"])[0]
        current_angle = deg(sensor_data)
        if (current_angle <= target_angle):
            self.set_propeller(module, ctrl)
            return False
        else:
            self.set_propeller(module, 0)
            return True
        
    def climb(self, module = "a", angle = 15):
        counter_module = "b" if module == "a" else "a"
        if (self.CLIMBSTEP == 0):
            self.fix_module(counter_module)
            self.release_module(module)
            h2_angled = self.set_h2_angle(module, angle)
            if (h2_angled):
                self.CLIMBSTEP = 1
        elif (self.CLIMBSTEP == 1):
            self.set_h2_angle(module, angle)
            h1_angled = self.set_h1_angle(counter_module, 90)
            if (h1_angled):
                self.CLIMBSTEP = 2
        elif (self.CLIMBSTEP == 2):
            self.fix_module(module)
            self.rotate_module(counter_module, 0)
            self.release_module(counter_module)
            self.CLIMBSTEP = 3
        elif (self.CLIMBSTEP == 3):
            at_height = self.set_height(counter_module, 5)
            if (at_height):
                self.CLIMBSTEP = 4
        elif (self.CLIMBSTEP == 4):
            h1_angled = self.set_h1_angle(module, 100)
            if (h1_angled):
                self.fix_module(counter_module)
                self.CLIMBSTEP = 5
        elif (self.CLIMBSTEP == 5):
            return True
        return False


    def transition_wall(self, module):
        if (self.TRANSITION_WALLSTEP == 0):
            model_a = self.set_distance("a", self.OBSTACLE_THRESHOLD)
            if (model_a):
                self.TRANSITION_WALLSTEP = 1
        elif (self.TRANSITION_WALLSTEP == 1):
            model_b = self.set_distance("b", self.OBSTACLE_THRESHOLD)
            if (model_b):
                self.TRANSITION_WALLSTEP = 2
        elif (self.TRANSITION_WALLSTEP == 2):
            climbed = self.climb(module)
            if (climbed):
                self.TRANSITION_WALLSTEP = 3
        elif (self.TRANSITION_WALLSTEP == 3):
            if (self.is_fixed("a") and self.is_fixed("b")):
                self.TRANSITION_WALLSTEP = 4
                return True 
        return False

