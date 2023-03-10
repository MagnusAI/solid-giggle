import mujoco

class LappaRobot:
    def __init__(self, xml_path):
        self.mj_model = mujoco.load_model_from_path(xml_path)
        self.mj_data = self.mj_model.data
        self.controller = None

    def set_controller(self, controller):
        self.controller = controller

    def simulate(self, dt):
        # This method advances the simulation by dt seconds
        self.mj_data.time += dt
        mujoco.forward(self.mj_model, self.mj_data)
        if self.controller is not None:
            self.controller.control(self.mj_data)
        mujoco.integrate(self.mj_model, self.mj_data, dt)

    def get_joint_angles(self):
        # This method returns the current angles of all the joints in the robot
        return self.mj_data.qpos[:]

    def set_joint_angles(self, angles):
        # This method sets the angles of all the joints in the robot
        self.mj_data.qpos[:] = angles

    def get_joint_velocities(self):
        # This method returns the current velocities of all the joints in the robot
        return self.mj_data.qvel[:]

    def set_joint_torque(self, torque):
        # This method sets the torque of all the joints in the robot
        self.mj_data.qfrc_actuator[:] = torque