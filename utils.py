"""Shared, side-effect-free helpers used across the project.

Function-only module (no top-level execution). Imported by the env factory,
the agents, the evaluation harness, and the Colab notebook.
"""

import os
import json
import random

import numpy as np
import yaml


def silence_warnings():
    """Mute the noisy (harmless) deprecation/legacy-gym warnings on Colab.

    Call this BEFORE importing stable-baselines3 / highway-env / pygame so the
    filters are active when those packages emit their import-time warnings:

      * "Gym has been unmaintained since 2022…" — SB3's internal legacy-``gym``
        compat import (we use Gymnasium; the shim import is harmless).
      * ``pkg_resources`` / ``declare_namespace`` — pygame + google.colab.
      * ``datetime.utcnow()`` — Jupyter kernel internals.

    These are warnings, not errors; this only quiets the output.
    """
    import warnings
    import logging

    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message=r".*Gym has been unmaintained.*")
    warnings.filterwarnings("ignore", message=r".*pkg_resources.*")
    warnings.filterwarnings("ignore", message=r".*declare_namespace.*")
    logging.getLogger("gym").setLevel(logging.ERROR)


def load_config(path):
    """Load the single project YAML into a plain dict.

    Args:
        path: path to ``configs/highway.yaml``.

    Returns:
        dict: the parsed configuration.
    """
    with open(path, "r") as f:
        return yaml.safe_load(f)


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


def curve_dir(cfg, tag):
    """Folder for a model's training-curve logs (CSV + TensorBoard).

    ``<drive_root>/<metrics>/curves/<tag>/`` — created if missing. The SB3 logger
    writes ``progress.csv`` here so the notebooks can plot PPO-vs-DQN curves.
    """
    d = os.path.join(cfg["paths"]["drive_root"], cfg["paths"]["metrics"], "curves", tag)
    os.makedirs(d, exist_ok=True)
    return d


def save_mp4(frames, path, fps=10):
    """Write RGB ``frames`` to an MP4 cleanly (no imageio resize warning).

    Frames are padded (not resized) up to the next multiple of 16 and encoded
    with H.264 + ``yuv420p`` so the file plays in any browser/QuickTime and the
    "macro_block_size" warning never fires. Returns ``path``.
    """
    import imageio

    if not frames:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    arr = []
    for f in frames:
        f = np.asarray(f)
        if f.ndim == 2:                      # greyscale -> RGB
            f = np.stack([f] * 3, axis=-1)
        if f.shape[2] == 4:                  # RGBA -> RGB
            f = f[..., :3]
        arr.append(f.astype(np.uint8))

    h = max(f.shape[0] for f in arr)
    w = max(f.shape[1] for f in arr)
    H = ((h + 15) // 16) * 16
    W = ((w + 15) // 16) * 16
    padded = [
        np.pad(f, ((0, H - f.shape[0]), (0, W - f.shape[1]), (0, 0)), mode="constant")
        for f in arr
    ]

    imageio.mimsave(
        path, padded, fps=fps, codec="libx264", quality=8,
        macro_block_size=16, pixelformat="yuv420p",
        output_params=["-loglevel", "error"],
    )
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
