# import register from gym
from gym.envs.registration import register

register(
    id='lappa-v0',
    entry_point='robot_env:RobotEnv',
    max_episode_steps=2000,
)
