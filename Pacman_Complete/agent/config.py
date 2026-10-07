"""Per-experiment configuration.

The 3-ghost and 4-ghost agents share the same network, replay buffer,
environment and reward function; they differ only by the values collected
here. Each configuration also owns the layout of the auxiliary feature
vector, because the number of ghosts determines where the pellet counters
sit inside it.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Everything that distinguishes one training run from another."""

    #: Number of ghosts spawned in the maze (0-4).
    num_ghosts: int
    #: Identifier appended to log and checkpoint file names.
    model_id: str
    #: Directory holding the checkpoints and logs of this experiment.
    run_dir: str
    #: Checkpoint written every 100 episodes and at the end of training.
    checkpoint_name: str
    #: Weights the evaluation script starts from. Training overwrites
    #: ``checkpoint_name`` continuously, so the weights that produced the
    #: reported results are kept under a separate, stable name.
    eval_checkpoint_name: str
    #: File the evaluation script loads the policy weights from.
    policy_name: str
    #: Name of the per-episode score log, inside ``run_dir/json_log``.
    score_log_name: str
    #: Name of the win log, inside ``run_dir/json_log``.
    win_log_name: str
    #: Number of training episodes.
    episodes: int
    #: Transitions collected before the first gradient step.
    warmup_steps: int
    #: Steps without eating that end the episode with a penalty.
    starving_limit: int
    #: Value used to pad the ghost-mode slots when a ghost is missing.
    mode_padding: int
    #: Whether the auxiliary vector carries the distance to the closest pellet.
    include_pellet_distance: bool
    #: The 4-ghost run clears the ghost-presence flag immediately after
    #: computing it, so the reward branches guarded by that flag never run.
    #: Kept per configuration to reproduce the behaviour each published
    #: result was actually obtained with.
    force_no_ghost_branch: bool
    #: Whether the win log records the step count and wall-clock duration.
    detailed_win_log: bool
    #: Number of episodes played by the evaluation script.
    eval_episodes: int

    # -- Derived paths ----------------------------------------------------

    @property
    def log_dir(self) -> str:
        return f"{self.run_dir}/json_log"

    @property
    def checkpoint_path(self) -> str:
        return f"{self.run_dir}/{self.checkpoint_name}"

    @property
    def eval_checkpoint_path(self) -> str:
        return f"{self.run_dir}/{self.eval_checkpoint_name}"

    @property
    def policy_path(self) -> str:
        return f"{self.run_dir}/{self.policy_name}"

    @property
    def score_log_path(self) -> str:
        return f"{self.log_dir}/{self.score_log_name}"

    @property
    def win_log_path(self) -> str:
        return f"{self.log_dir}/{self.win_log_name}"

    # -- Auxiliary vector layout ------------------------------------------
    #
    # The auxiliary vector fed to the network alongside the image is laid out
    # as follows, with G = num_ghosts:
    #
    #   [0]            normalised score
    #   [1]            normalised remaining lives
    #   [2 .. 2+G-1]   current mode of each ghost
    #   [2+G]          ratio of normal pellets left
    #   [3+G]          ratio of power pellets left
    #   [4+G]          1 if any ghost is frightened, else 0
    #   [5+G]          distance to the closest pellet (4-ghost run only)
    #   [.. .. ]       distance from Pacman to each ghost (G values)

    @property
    def normal_pellet_index(self) -> int:
        return 2 + self.num_ghosts

    @property
    def power_pellet_index(self) -> int:
        return 3 + self.num_ghosts

    @property
    def aux_size(self) -> int:
        return 5 + 2 * self.num_ghosts + int(self.include_pellet_distance)


#: Three ghosts, the first configuration that reached a stable policy.
THREE_GHOSTS = AgentConfig(
    num_ghosts=3,
    model_id="ultimate",
    run_dir="IA_3_fantomes",
    checkpoint_name="dqn_pacman_ultimate_test.pth",
    eval_checkpoint_name="dqn_pacman_ultimate_test.pth",
    policy_name="policy_only.pth",
    score_log_name="score_log.json",
    win_log_name="win_log_ultimate.json",
    episodes=6000,
    warmup_steps=5000,
    starving_limit=260,
    mode_padding=0,
    include_pellet_distance=False,
    force_no_ghost_branch=False,
    detailed_win_log=False,
    eval_episodes=5,
)

#: Four ghosts, the final configuration presented at the defence.
FOUR_GHOSTS = AgentConfig(
    num_ghosts=4,
    model_id="6",
    run_dir="IA_4_fantomes",
    checkpoint_name="dqn_pacman_train_6.pth",
    eval_checkpoint_name="modele_final.pth",
    policy_name="policy_only.pth",
    score_log_name="score_log_train_6.json",
    win_log_name="test30fps.json",
    episodes=20000,
    warmup_steps=10000,
    starving_limit=100,
    mode_padding=-1,
    include_pellet_distance=True,
    force_no_ghost_branch=True,
    detailed_win_log=True,
    eval_episodes=100,
)

CONFIGS = {3: THREE_GHOSTS, 4: FOUR_GHOSTS}


def get_config(num_ghosts: int) -> AgentConfig:
    """Return the configuration for ``num_ghosts``, or raise ValueError."""
    try:
        return CONFIGS[num_ghosts]
    except KeyError:
        raise ValueError(
            f"no configuration for {num_ghosts} ghosts; available: {sorted(CONFIGS)}"
        ) from None
