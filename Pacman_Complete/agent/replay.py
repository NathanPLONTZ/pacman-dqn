"""Prioritised replay buffer with n-step returns."""

from collections import namedtuple, deque

import numpy as np
import torch

from agent.networks import device

Transition = namedtuple(
    'Transition',
    ('state', 'aux', 'action', 'reward', 'next_state', 'next_aux', 'done')
)


class PrioritizedReplayMemory:
    """Replay buffer sampling transitions proportionally to their TD error.

    Incoming transitions first go through an n-step window, so what is stored
    is a single transition carrying the discounted return over ``n_step``
    environment steps. Sampling probabilities are the stored priorities raised
    to ``alpha``; the importance-sampling weights returned alongside the batch
    correct the resulting bias, with ``beta`` annealed by the caller.
    """

    def __init__(self, capacity, alpha=0.6, n_step=3, gamma=0.99):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.alpha = alpha
        self.n_step = n_step
        self.gamma = gamma
        self.n_step_buffer = deque(maxlen=n_step)

    def push(self, transition):
        """Queue a (s, aux, a, r, s', aux', done) tuple and store the n-step one."""
        self.n_step_buffer.append(transition)
        if len(self.n_step_buffer) < self.n_step:
            return
        R = sum([self.gamma**i * self.n_step_buffer[i][3] for i in range(self.n_step)])
        s, aux, a, _, _, _, done = self.n_step_buffer[0]
        _, _, _, _, s_n, aux_n, done_n = self.n_step_buffer[-1]
        new_trans = (s, aux, a, R, s_n, aux_n, done_n)
        if len(self.buffer) < self.capacity:
            self.buffer.append(new_trans)
        else:
            self.buffer[self.pos] = new_trans
        self.priorities[self.pos] = self.priorities.max() if self.buffer else 1.0
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Return a batch, the sampled indices and the correction weights."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        # Before any priority has been set, fall back to uniform sampling.
        if prios.sum() == 0:
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = prios ** self.alpha
            probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        total = len(self.buffer)
        weights = (total * probs[indices])**(-beta)
        weights /= weights.max()
        batch = Transition(*zip(*samples))
        return batch, indices, torch.tensor(weights, dtype=torch.float32, device=device)

    def update_priorities(self, indices, td_errors):
        """Re-prioritise the sampled transitions from their new TD errors."""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-5

    def __len__(self):
        return len(self.buffer)
