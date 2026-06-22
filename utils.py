"""Shared, side-effect-free helpers used across the project.

Function-only module (no top-level execution). Imported by the env factory,
the agents, the evaluation harness, and the Colab notebook.
"""

import os
import json
import random
import copy

import numpy as np
import yaml


def load_config(path):
    """Load the single project YAML into a plain dict.

    Args:
        path: path to ``configs/highway.yaml``.

    Returns:
        dict: the parsed configuration.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


def deep_update(base, overrides):
    """Recursively merge ``overrides`` into a copy of ``base`` (non-mutating).

    Lets the notebook tweak a few keys (e.g. a smaller ``total_timesteps`` for a
    smoke test) without editing the YAML on disk.
    """
    out = copy.deepcopy(base)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_global_seeds(seed):
    """Seed Python, NumPy and (if available) PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def drive_path(cfg, key, *parts):
    """Build an absolute path under the Drive results root.

    Example:
        drive_path(cfg, "checkpoints", "ppo.zip")
        -> /content/drive/MyDrive/nesy-highway-driving/checkpoints/ppo.zip
    """
    root = cfg["paths"]["drive_root"]
    sub = cfg["paths"][key]
    path = os.path.join(root, sub, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def save_json(obj, path):
    """Write ``obj`` to ``path`` as pretty JSON, creating parent dirs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    return path


def _json_default(o):
    """Make NumPy scalars/arrays JSON-serialisable."""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")
