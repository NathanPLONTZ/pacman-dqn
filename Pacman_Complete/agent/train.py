"""Training loop: Dueling Double DQN with NoisyNets, PER and n-step returns.

Run from the ``Pacman_Complete`` directory:

    python -m agent.train --ghosts 4
"""

import argparse
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from agent.config import AgentConfig, get_config
from agent.environment import NUM_CHANNELS, PacmanEnvJSON
from agent.frame_stack import FrameStack
from agent.networks import DQNConvAux, device
from agent.replay import PrioritizedReplayMemory
from agent.training_logs import log_score

#: Observations stacked and fed to the network at once.
FRAME_STACK_SIZE = 4

LEARNING_RATE = 2.5e-4
MIN_LEARNING_RATE = 1e-6
BATCH_SIZE = 64
GAMMA = 0.99
N_STEP = 3
REPLAY_CAPACITY = 200_000
PER_ALPHA = 0.6
PER_BETA_START = 0.4
#: One gradient step every this many environment steps.
TRAIN_EVERY = 4
#: Environment steps between two target-network syncs.
TARGET_UPDATE = 5000
#: Episodes between two checkpoint writes.
CHECKPOINT_EVERY = 100
#: Window of the moving average printed next to each episode.
MOVING_AVERAGE_WINDOW = 20


def train(config: AgentConfig):
    """Train an agent for ``config.episodes`` episodes and checkpoint it."""
    env = FrameStack(PacmanEnvJSON(config), k=FRAME_STACK_SIZE)
    env.set_number_of_ghosts(config.num_ghosts)

    in_channels = NUM_CHANNELS * FRAME_STACK_SIZE
    n_actions = env.action_space.n
    episodes = config.episodes

    policy_net = DQNConvAux(in_channels, config.aux_size, n_actions).to(device)
    target_net = DQNConvAux(in_channels, config.aux_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    policy_net.train()
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=episodes, eta_min=MIN_LEARNING_RATE
    )
    memory = PrioritizedReplayMemory(
        capacity=REPLAY_CAPACITY, alpha=PER_ALPHA, n_step=N_STEP, gamma=GAMMA
    )

    frame_idx = 0
    win = 0
    last_rewards = deque(maxlen=MOVING_AVERAGE_WINDOW)

    for episode in range(1, episodes + 1):
        state_img, aux = env.reset_image()
        state = torch.tensor(state_img, dtype=torch.float32, device=device).unsqueeze(0)
        aux_v = torch.tensor(aux, dtype=torch.float32, device=device).unsqueeze(0)

        total_reward = 0
        done = False
        won = False
        step = 0
        loss_sum = 0.0
        qmean_sum = 0.0
        updates = 0

        while not done:
            frame_idx += 1
            step += 1

            # Exploration comes from the NoisyLinear layers, so the action is
            # always the greedy one under freshly sampled noise.
            policy_net.reset_noise()
            target_net.reset_noise()
            with torch.no_grad():
                q = policy_net(state, aux_v)
            action = q.argmax(dim=1).item()

            (next_img, next_aux), reward, done, won, _ = env.step_image(action, step, episode)
            next_state = torch.tensor(next_img, dtype=torch.float32, device=device).unsqueeze(0)
            next_aux_v = torch.tensor(next_aux, dtype=torch.float32, device=device).unsqueeze(0)

            memory.push((state, aux_v, action, reward, next_state, next_aux_v, done))

            state, aux_v = next_state, next_aux_v
            total_reward += reward

            if frame_idx % TRAIN_EVERY == 0 and len(memory) >= config.warmup_steps:
                # Anneal the importance-sampling exponent towards 1.
                beta = min(1.0, PER_BETA_START + frame_idx * (0.6 / (episodes * 1000)))
                batch, indices, weights = memory.sample(BATCH_SIZE, beta)
                states, auxs, actions, rewards, next_states, next_auxs, dones = batch

                states = torch.cat(states)
                auxs = torch.cat(auxs)
                actions = torch.tensor(actions, device=device).unsqueeze(1)
                rewards = torch.tensor(rewards, device=device).unsqueeze(1)
                next_states = torch.cat(next_states)
                next_auxs = torch.cat(next_auxs)
                dones = torch.tensor(dones, device=device).unsqueeze(1)

                # Double DQN: the online net picks the action, the target net
                # scores it, which removes most of the overestimation bias.
                with torch.no_grad():
                    next_actions = policy_net(next_states, next_auxs).argmax(1, keepdim=True)
                    q_next = target_net(next_states, next_auxs).gather(1, next_actions)
                    target = rewards + (GAMMA ** memory.n_step) * q_next * (1 - dones.float())

                q_vals = policy_net(states, auxs).gather(1, actions)
                loss = (F.smooth_l1_loss(q_vals, target, reduction='none')
                        * weights.unsqueeze(1)).mean()

                loss_sum += loss.item()
                qmean_sum += q_vals.mean().item()
                updates += 1

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
                optimizer.step()

                td_errors = (q_vals - target).detach().cpu().squeeze().numpy()
                memory.update_priorities(indices, td_errors)

            if frame_idx % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

        scheduler.step()

        mean_loss = loss_sum / updates if updates > 0 else 0.0
        mean_q = qmean_sum / updates if updates > 0 else 0.0
        log_score(config.score_log_path, episode, total_reward, step, mean_q, mean_loss)

        last_rewards.append(total_reward)
        moving_avg = sum(last_rewards) / len(last_rewards)

        print(
            f"Episode {episode:3d} | steps {step:4d} | reward {total_reward:6.4f} "
            f"| MA{MOVING_AVERAGE_WINDOW}_reward {moving_avg:6.4f} "
            f"| mean Q {mean_q:.4f} | loss {mean_loss:.4f}"
            f" | lr {scheduler.optimizer.param_groups[0]['lr']:.6f}"
        )

        if won:
            win += 1
        if episode % CHECKPOINT_EVERY == 0:
            torch.save(policy_net.state_dict(), config.checkpoint_path)
            print(f"[WIn] Ghosts: {win}")

    torch.save(policy_net.state_dict(), config.checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghosts", type=int, default=4, choices=(3, 4),
                        help="which ghost configuration to train (default: 4)")
    args = parser.parse_args()
    train(get_config(args.ghosts))


if __name__ == '__main__':
    main()
