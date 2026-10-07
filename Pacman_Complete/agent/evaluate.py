"""Play a trained policy and report its win rate.

Run from the ``Pacman_Complete`` directory:

    python -m agent.evaluate --ghosts 4 --episodes 100
"""

import argparse
import os

import torch

from agent.config import AgentConfig, get_config
from agent.environment import NUM_CHANNELS, PacmanEnvJSON
from agent.frame_stack import FrameStack
from agent.networks import DQNConvAux
from agent.train import FRAME_STACK_SIZE

#: Evaluation runs on the GPU, as the reported results were produced.
EVAL_DEVICE = torch.device('cuda')


def extract_and_save_policy_only(full_ckpt_path, policy_only_path):
    """Write the policy weights alone to ``policy_only_path``.

    Training checkpoints may be plain state dicts or a dict holding the policy
    under a ``policy`` key; both are accepted. An existing file is left alone,
    so a policy extracted once is reused on later runs.
    """
    if not os.path.exists(policy_only_path):
        print(f"-> Extracting the policy from {full_ckpt_path}")
        cp = torch.load(full_ckpt_path, map_location="cpu")
        policy_state = cp.get('policy', cp)
        torch.save(policy_state, policy_only_path)
        print(f"Policy-only weights saved to {policy_only_path}")
    else:
        print(f"-> Policy-only weights already present: {policy_only_path}")


def run_trained_agent(config: AgentConfig, nb_episodes=None, render=False):
    """Play ``nb_episodes`` greedy episodes and print the resulting win rate."""
    nb_episodes = nb_episodes or config.eval_episodes
    env = FrameStack(PacmanEnvJSON(config), k=FRAME_STACK_SIZE)

    extract_and_save_policy_only(config.eval_checkpoint_path, config.policy_path)

    model = DQNConvAux(
        in_channels=NUM_CHANNELS * FRAME_STACK_SIZE,
        aux_size=config.aux_size,
        n_actions=env.action_space.n,
    ).to(EVAL_DEVICE)
    model.load_state_dict(torch.load(config.policy_path, map_location="cuda"))
    # eval() also switches the NoisyLinear layers to their mean weights, which
    # makes the policy deterministic.
    model.eval()

    win = 0
    for episode in range(nb_episodes):
        print(f"\nStarting episode {episode + 1}")

        obs_img, obs_aux = env.reset_image()
        state_img = torch.tensor(obs_img, dtype=torch.float32).unsqueeze(0)
        state_aux = torch.tensor(obs_aux, dtype=torch.float32).unsqueeze(0)

        done = False
        won = False
        total_reward = 0
        step = 0

        while not done:
            step += 1
            with torch.no_grad():
                q_values = model(state_img, state_aux)
                action = q_values.argmax(dim=1).item()
            (next_img, next_aux), reward, done, won, _ = env.step_image(action, step, episode + 1)
            total_reward += reward

            state_img = torch.tensor(next_img, dtype=torch.float32).unsqueeze(0)
            state_aux = torch.tensor(next_aux, dtype=torch.float32).unsqueeze(0)

            if render:
                env.render()

        outcome = "won" if won else "lost"
        if won:
            win += 1
        print(f"Episode {episode + 1} {outcome} with total reward {total_reward} "
              f"and used {step} steps")

    print(f"Total wins: {win}, win rate {(win / nb_episodes) * 100}%")
    env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghosts", type=int, default=4, choices=(3, 4),
                        help="which ghost configuration to evaluate (default: 4)")
    parser.add_argument("--episodes", type=int, default=None,
                        help="number of episodes to play (default: the config value)")
    parser.add_argument("--render", action="store_true",
                        help="call the environment render hook on every step")
    args = parser.parse_args()
    run_trained_agent(get_config(args.ghosts), args.episodes, args.render)


if __name__ == "__main__":
    main()
