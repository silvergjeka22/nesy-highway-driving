"""Plots for the notebooks: training curves, eval comparison, rule violations.

Function-only. Reads the CSV training logs written by ``agents.baselines``
(``metrics/curves/<tag>/progress.csv``) and the metric dicts returned by
``eval.evaluate.evaluate``. Every figure is returned (for inline display) and
optionally saved to Drive as a PNG. The notebooks call these.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import curve_dir, drive_path


# =============================================================================
# Part 1 — training curves (RecurrentPPO vs DQN vs QR-DQN)
# =============================================================================
def load_curve(cfg, tag):
    """Load one model's training curve from SB3's ``progress.csv``.

    Returns a frame with columns ``t`` (timesteps), ``rew`` (ep_rew_mean),
    ``len`` (ep_len_mean), ``overtakes``, ``lane_changes`` and ``crash`` (recorded
    by the training progress callback), or ``None`` if the log is missing.
    """
    csv = os.path.join(curve_dir(cfg, tag), "progress.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    cols = {"t": "time/total_timesteps", "rew": "rollout/ep_rew_mean",
            "len": "rollout/ep_len_mean", "overtakes": "rollout/ep_overtakes_mean",
            "lane_changes": "rollout/ep_lane_changes_mean",
            "crash": "rollout/ep_crash_rate"}
    out = pd.DataFrame()
    for k, c in cols.items():
        out[k] = df[c] if c in df.columns else np.nan
    return out.dropna(subset=["t"])


def plot_training_curves(cfg, tags=("rppo", "dqn", "qrdqn"), save=True):
    """Reward, episode length, overtakes, lane changes and crash rate vs timesteps.

    The overtakes panel shows the agent *learning to pass traffic*, the
    lane-changes panel whether it actually uses the manoeuvre actions, and the
    crash panel what the aggression costs. Returns the figure; saves
    ``metrics/training_curves.png`` to Drive.
    """
    panels = [("rew", "mean episode reward"), ("len", "mean episode length"),
              ("overtakes", "overtakes per episode"),
              ("lane_changes", "lane changes per episode"), ("crash", "crash rate")]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7))
    for tag in tags:
        df = load_curve(cfg, tag)
        if df is None or df.empty:
            continue
        for ax, (col, _) in zip(axes.flat, panels):
            ax.plot(df["t"], df[col], marker=".", label=tag.upper())
    for ax, (col, title) in zip(axes.flat, panels):
        ax.set(title=f"Training: {title}", xlabel="timesteps", ylabel=col)
        ax.grid(alpha=0.3)
        ax.legend()
    for ax in axes.flat[len(panels):]:
        ax.axis("off")
    fig.tight_layout()
    if save:
        fig.savefig(drive_path(cfg, "metrics", "training_curves.png"), dpi=120, bbox_inches="tight")
    return fig


# =============================================================================
# Parts 1-4 — evaluation comparison (bars with std)
# =============================================================================
def plot_eval_comparison(metrics_by_name, cfg=None, save_as=None):
    """Grouped bars comparing models on the headline eval metrics.

    Args:
        metrics_by_name: ``{label: evaluate(...) dict}``.
        cfg, save_as: if both given, save the PNG to ``metrics/<save_as>``.
    """
    names = list(metrics_by_name)
    panels = [
        ("crash_rate", lambda s: s["crash_rate"], None),
        ("distance (m)", lambda s: s["distance"]["mean"], lambda s: s["distance"]["std"]),
        ("overtakes (mean)", lambda s: s["overtakes"]["mean"], lambda s: s["overtakes"]["std"]),
        ("lane changes (mean)", lambda s: s["lane_changes"]["mean"], lambda s: s["lane_changes"]["std"]),
        ("return (mean)", lambda s: s["return"]["mean"], lambda s: s["return"]["std"]),
        ("on_road %", lambda s: s["on_road_pct"]["mean"], lambda s: s["on_road_pct"]["std"]),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    x = np.arange(len(names))
    for ax, (title, val, err) in zip(axes, panels):
        summ = [metrics_by_name[n]["summary"] for n in names]
        vals = [val(s) for s in summ]
        errs = [err(s) for s in summ] if err else None
        ax.bar(x, vals, yerr=errs, capsize=4, color="#1C7293")
        ax.set(title=title, xticks=x)
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    if cfg is not None and save_as:
        fig.savefig(drive_path(cfg, "metrics", save_as), dpi=120, bbox_inches="tight")
    return fig


# =============================================================================
# Part 2 (XAI) — per-rule violation rates: shield vs no-shield, etc.
# =============================================================================
RULES = ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2")


def plot_violation_rates(metrics_by_name, cfg=None, save_as="violation_rates.png"):
    """Grouped bars: per-rule violation rate for each config (the XAI headline).

    Lower is better. Use to show shield / logic-reward cut violations vs baseline.
    """
    names = list(metrics_by_name)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(RULES))
    width = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        rv = metrics_by_name[n]["summary"].get("rule_violation_rate", {})
        vals = [rv.get(r, 0.0) for r in RULES]
        ax.bar(x + i * width, vals, width, label=n)
    ax.set(title="Per-rule violation rate (independent monitor) — lower is better",
           xlabel="rule", ylabel="fraction of steps violating")
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(RULES)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    if cfg is not None and save_as:
        fig.savefig(drive_path(cfg, "metrics", save_as), dpi=120, bbox_inches="tight")
    return fig
