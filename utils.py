import os
import json
import random

import numpy as np
import yaml


def load_config(path):
    """Load the YAML config file that holds all hyperparameters and paths."""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def set_global_seeds(seed):
    """Pin Python, NumPy and PyTorch RNGs for reproducibility."""
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
    """Build an absolute path under the Drive results root, creating parents."""
    root = cfg["paths"]["drive_root"]
    sub = cfg["paths"][key]
    path = os.path.join(root, sub, *parts)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def curve_dir(cfg, tag):
    """Return (and create) the directory for a training curve CSV."""
    d = os.path.join(cfg["paths"]["drive_root"], cfg["paths"]["metrics"], "curves", tag)
    os.makedirs(d, exist_ok=True)
    return d


def save_mp4(frames, path, fps=10):
    """Encode a list of RGB arrays into an H.264 MP4. Pads to mod-16 for codec."""
    import imageio

    if not frames:
        return path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    arr = []
    for f in frames:
        f = np.asarray(f)
        if f.ndim == 2:
            f = np.stack([f] * 3, axis=-1)
        if f.shape[2] == 4:
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


def stitch_videos_grid(video_paths, path, fps=10, cols=None, downscale=2):
    """Combine multiple MP4s into a labelled side-by-side grid video."""
    import imageio

    clips, names = [], []
    for name, p in video_paths.items():
        if not (p and os.path.exists(p)):
            continue
        reader = imageio.get_reader(p)
        meta = reader.get_meta_data()
        n_read = int(round(meta.get("duration", 0) * meta.get("fps", fps))) or 1
        frames = []
        for i in range(n_read):
            try:
                frames.append(np.asarray(reader.get_data(i))[::downscale, ::downscale, :3])
            except (IndexError, RuntimeError):
                break
        reader.close()
        if frames:
            clips.append(frames)
            names.append(name)
    if not clips:
        raise ValueError("no readable clips in video_paths")

    cols = cols or len(clips)
    h = max(c[0].shape[0] for c in clips)
    w = max(c[0].shape[1] for c in clips)
    T = max(len(c) for c in clips)
    rows = (len(clips) + cols - 1) // cols

    def label(img, text):
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            return img
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 14 + 8 * len(text), 22], fill=(0, 0, 0))
        d.text((7, 5), text, fill=(255, 255, 255))
        return np.asarray(im)

    def tile(clip, t, name):
        f = clip[min(t, len(clip) - 1)]
        out = np.zeros((h, w, 3), np.uint8)
        out[: f.shape[0], : f.shape[1]] = f
        return label(out, name.upper())

    blank = np.zeros((h, w, 3), np.uint8)
    grid_frames = []
    for t in range(T):
        tiles = [tile(c, t, n) for c, n in zip(clips, names)]
        tiles += [blank] * (rows * cols - len(tiles))
        grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
        grid_frames.append(grid)
    return save_mp4(grid_frames, path, fps=fps)


def save_json(obj, path):
    """Write a dict to JSON, converting numpy types automatically."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=json_default)
    return path


def json_default(o):
    """JSON serializer fallback for numpy scalars and arrays."""
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Not JSON serialisable: {type(o)}")
