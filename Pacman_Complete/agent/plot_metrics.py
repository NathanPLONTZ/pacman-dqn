"""Plot the per-episode training metrics stored in a score log.

Run from the ``Pacman_Complete`` directory:

    python -m agent.plot_metrics --ghosts 4
"""

import argparse

import json
import matplotlib.pyplot as plt
import pandas as pd

from agent.config import get_config

#: Keys plotted from the score log. ``learning_rate`` holds the mean Q value
#: of the episode, see agent.training_logs.log_score.
METRICS = ['score', 'loss', 'learning_rate', 'step']


def plot_all_metrics_with_ma(json_log_path, window=20):
    """Plot every metric of a score log next to its moving average.

    Args:
        json_log_path: path to a score log written by agent.training_logs.
        window: size of the moving-average window, in episodes.
    """
    with open(json_log_path, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data).sort_values('episode')

    for m in METRICS:
        df[f'{m}_ma{window}'] = df[m].rolling(window=window, min_periods=1).mean()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, m in zip(axes, METRICS):
        ax.plot(df['episode'], df[m], label=m)
        ax.plot(df['episode'], df[f'{m}_ma{window}'], label=f'{m} MA({window})')
        ax.set_title(f'{m.capitalize()} per episode')
        ax.set_xlabel('Episode')
        ax.set_ylabel(m.capitalize())
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghosts", type=int, default=4, choices=(3, 4),
                        help="which run to plot (default: 4)")
    parser.add_argument("--window", type=int, default=20,
                        help="moving-average window, in episodes (default: 20)")
    args = parser.parse_args()
    plot_all_metrics_with_ma(get_config(args.ghosts).score_log_path, args.window)


if __name__ == "__main__":
    main()
