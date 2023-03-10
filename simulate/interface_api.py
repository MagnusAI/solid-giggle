class InterfaceLappaApi:
    def update_state(self, state):
        # Update the state of the robot (sensor values, actuator values, etc.)
        pass

    def read_sensors(self, module, sensor_names):
        # Read the sensor values of the robot
        # module: "a" or "b"
        # sensor_names: Array of sensor names
        pass

    def read_actuators(self, module, actuator_names):
        # Read the actuator values of the robot
        # module: "a" or "b"
        # actuator_names: Array of actuator names
        pass

    def set_actuator(self, module, actuator, ctrl):
        # Set the actuator value of the robot
        # module: "a" or "b"
        # actuator: Actuator name
        # ctrl: Control value
        pass

    def is_fixable(self, module):
        # Check if the module is fixable/above a surface and able to attach to it
        # module: "a" or "b"
        # return: True if the module is fixable, False otherwise
        pass

    def fix_module(self, module):
        # Fix the module to the surface
        # module: "a" or "b"
        # return: True if the module is fixed, False otherwise
        pass

    def release_module(self, module, ctrl = 0):
        # Release the module from the surface
        # module: "a" or "b"
        # ctrl: Control value to whihc the module is will rest at after release
        # return: True if the module is released, False otherwise
        pass

    def is_obstructed(self, module):
        # Check if the module is obstructed by something/an obstacle is in front of it
        # module: "a" or "b"
        # return: True if the module is obstructed, False otherwise
        pass

    def rotate_module(self, module, degrees, ctrl):
        # Rotate the module by the specified degrees with the specified control value
        # module: "a" or "b"
        # degrees: Degrees to rotate
        # ctrl: Control value
        # return: True if the module is rotated, False otherwise
        pass