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
# Part 1 — training curves (PPO vs DQN)
# =============================================================================
def load_curve(cfg, tag):
    """Load one model's training curve from SB3's ``progress.csv``.

    Returns a frame with columns ``t`` (timesteps), ``rew`` (ep_rew_mean) and
    ``len`` (ep_len_mean), or ``None`` if the log is missing.
    """
    csv = os.path.join(curve_dir(cfg, tag), "progress.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    cols = {"t": "time/total_timesteps", "rew": "rollout/ep_rew_mean", "len": "rollout/ep_len_mean"}
    out = pd.DataFrame()
    for k, c in cols.items():
        out[k] = df[c] if c in df.columns else np.nan
    return out.dropna(subset=["t"])


# Consistent colors so PPO and DQN look the same in every plot.
_ALGO_COLORS = {"ppo": "#1f77b4", "dqn": "#ff7f0e"}


def _smooth(series, window):
    """Rolling mean (centered) for a readable trend line; passthrough if too short."""
    if window <= 1 or len(series) < 3:
        return series
    return series.rolling(window, min_periods=1, center=True).mean()


def plot_training_curves(cfg, tags=("ppo", "dqn"), save=True):
    """Plot smoothed mean episode reward + length vs timesteps, PPO vs DQN.

    Each curve shows a rolling-mean trend line with the raw values faint behind it.
    Returns the figure; saves ``metrics/training_curves.png`` to Drive.
    """
    from matplotlib.ticker import FuncFormatter

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    panels = [("rew", "Mean episode reward", "ep_rew_mean"),
              ("len", "Mean episode length", "ep_len_mean")]

    plotted = False
    for tag in tags:
        df = load_curve(cfg, tag)
        if df is None or df.empty:
            continue
        plotted = True
        color = _ALGO_COLORS.get(tag.lower(), None)
        window = max(1, len(df) // 15)   # adapt smoothing to how many points we have
        for ax, (col, _title, _yl) in zip(axes, panels):
            ax.plot(df["t"], df[col], color=color, alpha=0.18, linewidth=1)          # raw, faint
            ax.plot(df["t"], _smooth(df[col], window), color=color, linewidth=2.4,   # smoothed trend
                    label=tag.upper())

    kfmt = FuncFormatter(lambda x, _pos: f"{x/1000:g}k" if x >= 1000 else f"{x:g}")
    for ax, (_col, title, ylabel) in zip(axes, panels):
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("timesteps", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.xaxis.set_major_formatter(kfmt)
        ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.4)
        ax.margins(x=0.01)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        if plotted:
            ax.legend(title="algorithm", frameon=True, fontsize=10)

    fig.suptitle("Training curves — PPO vs DQN", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save:
        fig.savefig(drive_path(cfg, "metrics", "training_curves.png"), dpi=130, bbox_inches="tight")
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
        ("return (mean)", lambda s: s["return"]["mean"], lambda s: s["return"]["std"]),
        ("overtakes (mean)", lambda s: s["overtakes"]["mean"], lambda s: s["overtakes"]["std"]),
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
