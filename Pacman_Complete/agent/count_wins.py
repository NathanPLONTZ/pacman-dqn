"""Count the victories recorded in a win log.

Each entry of a win log is one cleared maze, so the number of wins is the
number of entries. Note that a log accumulates across runs: the evaluation
script appends to the existing file instead of starting a new one.

Run from the ``Pacman_Complete`` directory:

    python -m agent.count_wins --ghosts 4
"""

import argparse
import json

from agent.config import get_config


def count_wins(json_file_path):
    """Return the number of victories recorded in a win log."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return len(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ghosts", type=int, default=4, choices=(3, 4),
                        help="which run to read (default: 4)")
    args = parser.parse_args()
    path = get_config(args.ghosts).win_log_path
    print(f"Number of victories in {path}: {count_wins(path)}")


if __name__ == "__main__":
    main()
