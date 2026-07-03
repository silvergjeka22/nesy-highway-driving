"""Replay buffers for Part 1.

ReplayBuffer          — plain uniform ring buffer (real environment data).
LifetimeReplayBuffer  — FIFO buffer for model-generated data, replicating the
                        paper's ReplayBufferDynamicLifeTime: every rollout round
                        the data of rounds older than `lifetime` rounds is
                        expired, so the policy only ever trains on imagined
                        transitions produced by a recent model/policy pair.
"""
import numpy as np
import torch


def _to_tensors(arrays, idx, device):
    out = []
    for arr in arrays:
        t = torch.from_numpy(arr[idx])
        if device.type == "cuda":
            t = t.pin_memory().to(device, non_blocking=True)
        else:
            t = t.to(device)
        out.append(t)
    return tuple(out)


class ReplayBuffer:
    """Uniform ring buffer storing (s, a, r, s', done)."""

    def __init__(self, capacity: int, state_dim: int, action_dim: int):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self._ptr  = 0
        self._size = 0

        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self._ptr]      = state
        self.actions[self._ptr]     = action
        self.rewards[self._ptr]     = reward
        self.next_states[self._ptr] = next_state
        self.dones[self._ptr]       = float(done)
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self._size, size=batch_size)
        return _to_tensors(
            (self.states, self.actions, self.rewards, self.next_states, self.dones),
            idx, device,
        )

    def sample_states(self, n: int) -> np.ndarray:
        idx = np.random.randint(0, self._size, size=min(n, self._size))
        return self.states[idx]

    def sample_all(self):
        return (self.states[:self._size], self.actions[:self._size],
                self.rewards[:self._size].squeeze(-1),
                self.next_states[:self._size], self.dones[:self._size])

    def __len__(self):
        return self._size


class LifetimeReplayBuffer:
    """FIFO buffer whose entries expire after `lifetime` rollout rounds.

    Mirrors the official MACURA repo's ReplayBufferDynamicLifeTime: the model
    buffer must only contain transitions imagined by the last
    `retain_epochs × trains_per_epoch` model/policy snapshots, even when
    uncertainty truncation makes rounds add far fewer than capacity/lifetime
    transitions (a plain ring buffer would let stale data linger).
    """

    def __init__(self, capacity: int, state_dim: int, action_dim: int, lifetime: int):
        self.capacity   = capacity
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.lifetime   = max(1, int(lifetime))
        self._tail = 0          # index of oldest element
        self._size = 0
        self._round_sizes = []  # transitions added per round, oldest first

        self.states      = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1),          dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim),  dtype=np.float32)
        self.dones       = np.zeros((capacity, 1),          dtype=np.float32)

    # ── Round lifecycle ────────────────────────────────────────────────────────

    def begin_round(self):
        """Open a new rollout round and expire rounds older than `lifetime`."""
        self._round_sizes.append(0)
        while len(self._round_sizes) > self.lifetime:
            expired = self._round_sizes.pop(0)
            expired = min(expired, self._size)
            self._tail = (self._tail + expired) % self.capacity
            self._size -= expired

    def add_batch(self, states, actions, rewards, next_states, dones):
        n = len(states)
        if n == 0:
            return
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if dones.ndim == 1:
            dones = dones[:, None]
        if n > self.capacity:        # keep only the newest items
            states, actions, rewards = (states[-self.capacity:],
                                        actions[-self.capacity:],
                                        rewards[-self.capacity:])
            next_states, dones = next_states[-self.capacity:], dones[-self.capacity:]
            n = self.capacity

        # Account the new items to the current round BEFORE evicting, so a
        # batch that overflows capacity can evict from its own round too.
        if self._round_sizes:
            self._round_sizes[-1] += n
        else:
            self._round_sizes = [n]
        overflow = self._size + n - self.capacity
        if overflow > 0:
            self._evict(overflow)

        head = (self._tail + self._size) % self.capacity
        idx  = (head + np.arange(n)) % self.capacity
        self.states[idx]      = states
        self.actions[idx]     = actions
        self.rewards[idx]     = rewards
        self.next_states[idx] = next_states
        self.dones[idx]       = dones
        self._size += n

    def _evict(self, count: int):
        """Drop the `count` oldest stored transitions and keep the per-round
        bookkeeping consistent."""
        count = min(count, self._size)
        self._tail = (self._tail + count) % self.capacity
        self._size -= count
        remaining = count
        while remaining > 0 and self._round_sizes:
            take = min(self._round_sizes[0], remaining)
            self._round_sizes[0] -= take
            remaining -= take
            if self._round_sizes[0] == 0:
                if len(self._round_sizes) > 1:
                    self._round_sizes.pop(0)
                else:
                    break

    # ── Sampling ───────────────────────────────────────────────────────────────

    def sample(self, batch_size: int, device: torch.device):
        offsets = np.random.randint(0, self._size, size=batch_size)
        idx = (self._tail + offsets) % self.capacity
        return _to_tensors(
            (self.states, self.actions, self.rewards, self.next_states, self.dones),
            idx, device,
        )

    def __len__(self):
        return self._size
