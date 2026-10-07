"""Frame stacking wrapper.

A single maze image says where everything is but not where it is going. The
network is therefore fed the last ``k`` observations concatenated along the
channel axis, which makes ghost and Pacman motion visible.
"""

from collections import deque

import numpy as np


class FrameStack:
    """Stack the ``k`` most recent observations of a PacmanEnvJSON."""

    def __init__(self, env, k=4):
        self.env = env
        self.k = k
        self.frames = deque(maxlen=k)

    def reset_image(self):
        img, aux = self.env.reset_image()
        # Prime the deque with k copies of the first frame.
        for _ in range(self.k):
            self.frames.append(img)
        return self._get_obs(), aux

    def step_image(self, action, *args, **kwargs):
        (img, aux), reward, done, won, info = self.env.step_image(action, *args, **kwargs)
        self.frames.append(img)
        return (self._get_obs(), aux), reward, done, won, info

    def _get_obs(self):
        """Concatenate the stored frames into a (k * channels, H, W) array."""
        return np.concatenate(list(self.frames), axis=0)

    def set_number_of_ghosts(self, n):
        self.env.phantome(n)

    def render(self, mode="human"):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

    @property
    def action_space(self):
        return self.env.action_space
