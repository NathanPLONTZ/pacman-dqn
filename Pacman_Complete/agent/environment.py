"""Gym environment wrapping the pygame Pacman game.

Observations are built directly from the live game objects rather than from
the rendered pixels: an image-like array describing the maze, plus a short
auxiliary vector holding the quantities a single frame cannot convey (score,
lives, ghost modes and distances).
"""

import json
import os
import time

import gym
import numpy as np
from gym import spaces

from agent.config import AgentConfig
from agent.game_bridge import GameStateExporter
from agent.maze import CELL_SIZE, get_wall_pixel_positions

#: Maze dimensions, in tiles.
GRID_ROWS, GRID_COLS = 36, 28

#: Observation channels: the full maze, then the window centred on Pacman.
NUM_CHANNELS = 2

#: Cell encoding of the observation image.
WALL_VALUE = -0.5
PELLET_VALUE = 0.5
POWER_PELLET_VALUE = 0.8
GHOST_VALUE = -1.0
PACMAN_VALUE = 1.0

#: The whole image is multiplied by this factor to move the values away from
#: zero before they reach the first convolution.
VALUE_SCALE = 30

#: Side of the local window copied into the second channel, in tiles.
LOCAL_WINDOW = 14

#: Normalisation constants for the auxiliary vector.
MAX_SCORE = 4500
MAX_LIVES = 5
MAX_NORMAL_PELLETS = 236
MAX_POWER_PELLETS = 4

#: Distance assigned to a ghost slot that no ghost occupies.
MISSING_GHOST_DISTANCE = 50

#: Mode identifier of a frightened ghost, i.e. one Pacman can eat.
FREIGHT_MODE = 2

#: Number of aux slots the reward function scans for ghost presence. Fixed at
#: four, the maximum number of ghosts, regardless of how many are spawned.
GHOST_MODE_SLOTS = 4


class PacmanEnvJSON(gym.Env):
    """Single-maze Pacman environment exposing image + auxiliary observations."""

    def __init__(self, config: AgentConfig):
        super(PacmanEnvJSON, self).__init__()
        self.config = config
        self.exporter = GameStateExporter(config.num_ghosts)

        self.action_precdente = None
        self.action_loop = 0
        self.step_starving = 0
        self.step_super = 0
        self.super_ate = False
        self.super_ate_2 = False
        self.number_of_actions_used = 0
        self.start_time = 0

        # Up, down, left, right.
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(NUM_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32,
        )

        self.wall_layer = self._precompute_wall_layer()

    # -- Episode bookkeeping ----------------------------------------------

    def reset_internal_state(self):
        """Clear the counters the reward function carries across steps."""
        self.action_loop = 0
        self.step_starving = 0
        self.step_super = 0
        self.super_ate = False
        self.super_ate_2 = False
        self.action_precdente = None

    def _precompute_wall_layer(self):
        """Build the static wall channel once; walls never move."""
        wall_layer = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        for x, y in get_wall_pixel_positions("maze1.txt"):
            wall_layer[int(y // CELL_SIZE), int(x // CELL_SIZE)] = WALL_VALUE
        return wall_layer

    def phantome(self, nombre):
        """Change how many ghosts the next game start will spawn."""
        self.exporter.game.setNombrePhantome(nombre)

    # -- Observation -------------------------------------------------------

    def get_current_state_and_aux(self):
        """Return the (channels, 36, 28) image and the auxiliary vector.

        Channel 0 holds the whole maze: walls, pellets, ghosts and Pacman all
        written into the same plane. Channel 1 repeats the sub-window centred
        on Pacman, leaving the rest at zero, so the network always gets a
        close-up of the area it is about to move through.
        """
        config = self.config
        state_img = np.zeros((NUM_CHANNELS, GRID_ROWS, GRID_COLS), dtype=np.float32)
        state_img[0] = self.wall_layer

        game = self.exporter.game

        px, py = game.pacman.position.x, game.pacman.position.y
        ci, cj = int(py // CELL_SIZE), int(px // CELL_SIZE)

        # Pellets: the power pellets are a subset of the full pellet list.
        all_pellets = [(p.position.x, p.position.y) for p in game.pellets.pelletList]
        power_positions = [(p.position.x, p.position.y) for p in game.pellets.powerpellets]
        normal_positions = [pos for pos in all_pellets if pos not in power_positions]
        power_pellets = [pos for pos in all_pellets if pos in power_positions]
        normal_count = len(normal_positions) / MAX_NORMAL_PELLETS
        power_count = len(power_pellets) / MAX_POWER_PELLETS

        for (x, y) in normal_positions:
            state_img[0, int(y // CELL_SIZE), int(x // CELL_SIZE)] = PELLET_VALUE
        for (x, y) in power_pellets:
            state_img[0, int(y // CELL_SIZE), int(x // CELL_SIZE)] = POWER_PELLET_VALUE

        if config.include_pellet_distance:
            pellet_dist_norm = self._nearest_pellet_distance(
                normal_positions + power_pellets, px, py
            )

        # Ghosts: position on the image, mode and distance in the aux vector.
        modes = []
        distances = []
        invincible = 0
        for ghost in game.ghosts:
            gx, gy = ghost.position.x, ghost.position.y
            gm = ghost.mode.current
            if gm == FREIGHT_MODE:
                invincible = 1
            state_img[0, int(gy // CELL_SIZE), int(gx // CELL_SIZE)] = GHOST_VALUE
            distances.append(((gy - py)**2 + (gx - px)**2)**0.5)
            modes.append(gm)

        # Pacman is written last so it wins over a ghost sharing its tile.
        state_img[0, ci, cj] = PACMAN_VALUE

        # Keep the aux vector a fixed length even with fewer ghosts than slots.
        while len(modes) < config.num_ghosts:
            modes.append(config.mode_padding)
        while len(distances) < config.num_ghosts:
            distances.append(MISSING_GHOST_DISTANCE)

        state_img[1] = self._local_window(state_img[0], ci, cj)
        state_img[0] *= VALUE_SCALE
        state_img[1] *= VALUE_SCALE

        score_norm = game.score / MAX_SCORE
        lives_norm = game.lives / MAX_LIVES
        aux_values = [score_norm, lives_norm, *modes, normal_count, power_count, invincible]
        if config.include_pellet_distance:
            aux_values.append(pellet_dist_norm)
        aux_values.extend(distances)

        return state_img, np.array(aux_values, dtype=np.float32)

    @staticmethod
    def _nearest_pellet_distance(pellet_positions, px, py):
        """Euclidean distance to the closest pellet, normalised to [0, 1]."""
        if not pellet_positions:
            return 1.0  # No pellet left, treat it as the furthest possible.
        min_distance = min(
            (((x - px)**2 + (y - py)**2)**0.5) for (x, y) in pellet_positions
        )
        max_dist = ((GRID_COLS**2 + GRID_ROWS**2)**0.5) * CELL_SIZE
        return min_distance / max_dist

    @staticmethod
    def _local_window(full_layer, ci, cj):
        """Copy the window centred on Pacman into an otherwise empty grid."""
        half = LOCAL_WINDOW // 2
        window = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.float32)
        r0, r1 = max(0, ci - half), min(GRID_ROWS, ci + half + 1)
        c0, c1 = max(0, cj - half), min(GRID_COLS, cj + half + 1)
        window[r0:r1, c0:c1] = full_layer[r0:r1, c0:c1]
        return window

    # -- Gym interface -----------------------------------------------------

    def step_image(self, action, step, episode):
        """Apply one action and return (observation, reward, done, won, info)."""
        self.number_of_actions_used += 1
        if self.action_precdente == action:
            self.action_loop += 1
        else:
            self.action_loop = 0
        self.action_precdente = action

        prev_img, prev_aux = self.get_current_state_and_aux()
        self.exporter.step_image(action)
        new_img, new_aux = self.get_current_state_and_aux()

        reward, done, won = self.calculate_reward_image(
            prev_img, new_img, prev_aux, new_aux, step, episode
        )
        return (new_img, new_aux), reward, done, won, {}

    def reset_image(self):
        """Restart the game and return the first observation."""
        self.start_time = time.time()
        self.number_of_actions_used = 0
        self.exporter.reset_image()
        self.exporter.step_image(5)
        self.reset_internal_state()
        return self.get_current_state_and_aux()

    def render(self, mode="human"):
        """No-op: the game already draws itself to its own pygame window."""
        pass

    # -- Logging -----------------------------------------------------------

    def log_win(self, episode):
        """Append one entry to the win log of this configuration."""
        log_file = self.config.win_log_path
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        if os.path.exists(log_file):
            try:
                with open(log_file, "r") as f:
                    logs = json.load(f)
            except (json.JSONDecodeError, ValueError):
                logs = []
        else:
            logs = []

        entry = {"episode": int(episode)}
        if self.config.detailed_win_log:
            entry["step"] = int(self.number_of_actions_used)
            entry["time"] = int(time.time() - self.start_time)
        logs.append(entry)

        with open(log_file, "w") as f:
            json.dump(logs, f, indent=1)

    # -- Reward ------------------------------------------------------------

    def calculate_reward_image(self, prev_state, new_state, prev_aux, new_aux,
                               step, episode):
        """Score the transition and decide whether the episode is over.

        The shaping pushes the agent to clear the maze rather than to survive:
        eating is worth more as fewer pellets remain, standing still without
        eating eventually aborts the episode, losing a life is penalised, and
        clearing the maze pays a large terminal bonus. Everything is divided by
        ten at the end to keep the returns in a range the network handles well.
        """
        config = self.config
        won = False

        # A ghost slot holds -1 when no ghost occupies it; anything else means
        # at least one ghost is on the board.
        fantome = False
        for i in range(GHOST_MODE_SLOTS):
            if new_aux[2 + i] != -1:
                fantome = True

        done = False
        if config.force_no_ghost_branch:
            fantome = False

        # --- Pellets eaten during this transition ---
        prev_super = prev_aux[config.power_pellet_index] * MAX_POWER_PELLETS
        new_super = new_aux[config.power_pellet_index] * MAX_POWER_PELLETS
        super_eaten = prev_super - new_super

        if (super_eaten == 1) and (self.super_ate):
            self.super_ate_2 = True
        elif (super_eaten == 1) and (not self.super_ate):
            self.super_ate = True

        new_normal = new_aux[config.normal_pellet_index] * MAX_NORMAL_PELLETS
        prev_normal = prev_aux[config.normal_pellet_index] * MAX_NORMAL_PELLETS
        normal_eaten = prev_normal - new_normal

        # --- Lives and score before / after ---
        prev_lives = prev_aux[1] * MAX_LIVES
        new_lives = new_aux[1] * MAX_LIVES

        prev_score = prev_aux[0] * MAX_SCORE
        new_score = new_aux[0] * MAX_SCORE

        # Score gained beyond the pellets themselves: eaten ghosts and fruit.
        score = (new_score - prev_score) - (normal_eaten * 10) - (super_eaten * 30)

        reward = 0.0

        # Eating pays more as the maze empties, to push the last pellets.
        if normal_eaten > 0:
            self.step_starving = 0
            reward += (-95.0 / MAX_NORMAL_PELLETS) * new_normal + 100
        if super_eaten > 0:
            self.step_starving = 0
            reward += super_eaten * 100

        starving_limit = config.starving_limit
        if (self.step_starving <= starving_limit) and ((normal_eaten + super_eaten) == 0) and (not fantome):
            self.step_starving += 1

        if (self.super_ate) and (fantome):
            self.step_super += 1

        # Penalise a power pellet eaten without cashing in on the ghosts.
        if (self.step_super >= 50) and (fantome):
            if self.super_ate_2:
                reward -= 100
            self.step_super = 0
            self.super_ate = False
            self.super_ate_2 = False

        # Too long without eating: abort the episode.
        if (self.step_starving >= starving_limit) and ((normal_eaten + super_eaten) == 0) and (not fantome):
            reward -= 300
            done = True
            self.action_loop = 0
            self.step_starving = 0
            self.step_super = 0
            self.super_ate = False
            self.super_ate_2 = False

        if score > 0:
            reward += (score // 200) * 50

        if new_lives < prev_lives:
            reward -= 200.0

        # Fixed per-step cost, so dawdling is never free.
        reward -= 0.1

        # --- Terminal conditions ---
        if new_lives <= 0:
            self.action_loop = 0
            self.step_starving = 0
            self.step_super = 0
            self.super_ate = False
            self.super_ate_2 = False
            done = True
        if (new_super + new_normal) == 0:
            self.log_win(episode)
            reward += 2000.0
            won = True
            self.action_loop = 0
            self.step_starving = 0
            self.step_super = 0
            self.super_ate = False
            done = True
            self.super_ate_2 = False

        return reward / 10, done, won
