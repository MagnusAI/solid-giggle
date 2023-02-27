# MuJoCa Python API - mujoco_py
# This is a simple controller written in python for th Lappa model

def controller(model, data):
    #put the controller here
    data.actuator("a_thrust").ctrl = -.5
    data.actuator("b_thrust").ctrl = -.5

    def is_grounded(module):
        if (module == "a"):
            return data.sensor("a_rangefinder_down").data < 0.01
        elif (module == "b"):
            return data.sensor("b_rangefinder_down").data < 0.01
        else:
            print(is_grounded.__name__ + ": Invalid module name")

    def set_thrust(module, thrust):
        if (module == "a"):
            data.actuator("a_thrust").ctrl = thrust
        elif (module == "b"):
            data.actuator("b_thrust").ctrl = thrust
        else:
            print(set_thrust.__name__ + ": Invalid module name")

    def set_rotate(module, angle):
        if (module == "a"):
            data.actuator("a_rotor").ctrl = -.1
        elif (module == "b"):
            data.actuator("b_rotate").ctrl = angle
        else:
            print(set_rotate.__name__ + ": Invalid module name")

    # If the robot if on the ground, then print "Grounded!" else print "In Air!
    if is_grounded("a") and is_grounded("b"):
        set_thrust("a", -1)
        set_thrust("b", 0.5)
        set_rotate("a", 0)

    pass