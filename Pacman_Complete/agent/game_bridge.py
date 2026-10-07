"""Bridge between the agent and the pygame implementation of the game.

The controller is driven one action at a time instead of by the keyboard, and
each agent action is applied over several game frames so that Pacman actually
travels between two decisions.
"""

import os
import sys

# The game modules sit one directory above this package and import each other
# by bare module name, so the project root has to be importable.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from run import GameController

STOP = 0
UP = 1
DOWN = -1
LEFT = 2
RIGHT = -2

#: Agent action index -> direction understood by the game. Action 4 releases the
#: controls (no direction change); action 5 is the no-op used to let the game
#: settle after a reset.
ACTION_MAP = {0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: None, 5: 5}

#: Game frames advanced per agent action.
FRAMES_PER_ACTION = 5


class GameStateExporter:
    """Own a GameController and step it on behalf of the environment."""

    def __init__(self, num_ghosts):
        self.game = GameController(training=False)
        self.game.setNombrePhantome(num_ghosts)
        self.game.startGame()
        self.step_image(5)

    def step_image(self, action):
        """Apply one agent action over FRAMES_PER_ACTION game frames."""
        action = ACTION_MAP.get(action)
        for _ in range(FRAMES_PER_ACTION):
            self.game.update(action)

    def reset_image(self):
        """Restart the game from a fresh level with full lives and score."""
        self.game.restartGame()
