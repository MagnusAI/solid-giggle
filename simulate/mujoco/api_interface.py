class InterfaceLappaApi:
    def update_data(self, data):
        # Update the state of the robot (sensor values, actuator values, etc.)
        pass
    
    def set_thruster(self, module, ctrl):
        # Set the thruster value of the robot
        # module: "a" or "b"
        # ctrl: Control value
        pass
    
    def get_thruster(self, module):
        # Get the thruster value of the robot
        # module: "a" or "b"
        pass
    
    def set_bouding_force(self, module, value):
        # Set the bounding force value of the robot
        # module: "a" or "b"
        # value: Control value
        pass
    
    def get_bouding_force(self, module):
        # Get the bounding force value of the robot
        # module: "a" or "b"
        pass
    
    def get_h1(self, module):
        # Get the h1 value of the robot
        # module: "a" or "b"
        pass
    
    def get_h2(self, module):
        # Get the h2 value of the robot
        # module: "a" or "b"
        pass
    
    def rotate_module(self, module, ctrl):
        # Rotate the module by the specified degrees with the specified control value
        # module: "a" or "b"
        # ctrl: Control value
        # return: True if the module is rotated, False otherwise
        pass
    
    def reset(self):
        # Reset the to the initial state
        pass
    