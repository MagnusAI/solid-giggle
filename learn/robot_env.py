import gym
from gym import spaces
import numpy as np
import pygame


class RobotEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # Define the observation space
        self.observation_space = spaces.Box(
            low=np.zeros(8), high=np.ones(8), dtype=np.float32)

        # Define the action space
        self.action_space = spaces.Box(low=np.array([-1, -1, 0, 0]),
                                       high=np.array([1, 1, 360, 360]),
                                       dtype=np.float32)

        # Define other environment-specific parameters
        self.time_limit = 1000
        self.current_step = 0
        self.reward_goal = 100
        self.reward_step = -1

        # Define the initial state of the robot
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    def step(self, action):
        # Update the state of the robot based on the action
        self.state[2] = np.clip(self.state[2] + action[2], -45, 45)
        self.state[3] = np.clip(self.state[3] + action[3], -45, 45)
        self.state[4] = np.clip(self.state[4] + action[0], 0, 360)
        self.state[5] = np.clip(self.state[5] + action[1], 0, 360)

        # Calculate the reward based on the new state
        if (self.state[0] and self.state[1] and self.state[6] >= 0.5 and self.state[7] >= 0.5):
            reward = self.reward_goal
        else:
            reward = self.reward_step

        # Check if the episode is done
        done = (self.current_step >= self.time_limit)
        self.current_step += 1

        # Return the new state, reward, and done flag
        return self.state, reward, done, {}

    def reset(self):
        # Reset the environment to its initial state
        self.current_step = 0
        self.state = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        return self.state

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
