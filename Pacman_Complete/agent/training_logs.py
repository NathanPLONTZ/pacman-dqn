"""Append-only JSON logs written during training."""

import json
import os


def append_json_log(log_file, entry):
    """Append one record to a JSON array file, creating it if needed.

    The file is rewritten in full on every call, which keeps it valid JSON and
    readable by the plotting helpers even while training is still running. A
    file that cannot be parsed is restarted from scratch rather than aborting
    a run that may already be hours long.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    if os.path.exists(log_file):
        try:
            with open(log_file, "r") as f:
                logs = json.load(f)
        except (json.JSONDecodeError, ValueError):
            logs = []
    else:
        logs = []

    logs.append(entry)
    with open(log_file, "w") as f:
        json.dump(logs, f, indent=4)


def log_score(log_file, episode, score, step, mean_q, loss):
    """Record the outcome of one training episode.

    ``mean_q`` is stored under the ``learning_rate`` key: the plotting helpers
    and every log produced so far use that name, so it is kept for
    compatibility with the existing files.
    """
    append_json_log(log_file, {
        "episode": int(episode),
        "score": float(score),
        "loss": float(loss),
        "learning_rate": float(mean_q),
        "step": int(step),
    })
