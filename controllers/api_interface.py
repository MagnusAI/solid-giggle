import abc

class ILappaApi(abc.ABC):
    @abc.abstractmethod
    def is_locked(self):
        pass

    @abc.abstractmethod
    def lock(self):
        pass

    @abc.abstractmethod
    def unlock(self):
        pass

    @abc.abstractmethod
    def debug_info(self):
        pass

    @abc.abstractmethod
    def update_data(self, data):
        pass
    
    @abc.abstractmethod
    def get_data(self):
        pass

    @abc.abstractmethod
    def set_thruster(self, module, ctrl):
        pass

    @abc.abstractmethod
    def get_thruster(self, module):
        pass

    @abc.abstractmethod
    def set_adhesion(self, module, value):
        pass

    @abc.abstractmethod
    def get_adhesion(self, module):
        pass

    @abc.abstractmethod
    def get_h1_actuator(self, module):
        pass

    @abc.abstractmethod
    def get_range(self, module):
        pass

    @abc.abstractmethod
    def get_h1(self, module):
        pass

    @abc.abstractmethod
    def get_h2(self, module):
        pass

    @abc.abstractmethod
    def get_pressure(self, module):
        pass

    @abc.abstractmethod
    def stop_rotation(self, module):
        pass

    @abc.abstractmethod
    def rotate_module(self, module, ctrl):
        pass

    @abc.abstractmethod
    def reset_h1(self, module):
        pass

    @abc.abstractmethod
    def get_position(self, module):
        pass

    @abc.abstractmethod
    def lift(self, module):
        pass

    @abc.abstractmethod
    def lower(self, module):
        pass

    @abc.abstractmethod
    def rotate(self, module, ctrl):
        pass

    @abc.abstractmethod
    def rotate_forward(self, module):
        pass

    @abc.abstractmethod
    def rotate_backward(self, module):
        pass

    @abc.abstractmethod
    def update_pressure(self):
        pass

    @abc.abstractmethod
    def perform_action(self, action):
        pass

    @abc.abstractmethod
    def get_arm_angle(self):
        pass

    @abc.abstractmethod
    def get_distance(self, module):
        pass

    @abc.abstractmethod
    def get_state(self):
        pass
