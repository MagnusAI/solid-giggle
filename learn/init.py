import gym
import robot_env

import gym
from gym.envs.registration import register

register(
    id='robot_env-v0',
    entry_point='robot_env:RobotEnv',
)

env = gym.make('robot_env-v0')
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print('Observation:', obs)
    print('Reward:', reward)
    print('Done:', done)
    print('Info:', info)
env.close()