# import register from gym
from gymnasium.envs.registration import register

register(
     id="gym_lappa/MyHalfCheetah-v0",
     entry_point="gym_lappa.envs:HalfCheetahEnv",
     max_episode_steps=1000,
    reward_threshold=4800.0,
)
