import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils import curve_dir, drive_path


def load_curve(cfg, tag):
    """Load a training curve CSV into a DataFrame with standardised column names."""
    csv = os.path.join(curve_dir(cfg, tag), "progress.csv")
    if not os.path.exists(csv):
        return None
    df = pd.read_csv(csv)
    cols = {"t": "time/total_timesteps", "rew": "rollout/ep_rew_mean",
            "len": "rollout/ep_len_mean", "overtakes": "rollout/ep_overtakes_mean",
            "lane_changes": "rollout/ep_lane_changes_mean",
            "crash": "rollout/ep_crash_rate",
            "viol_RI2": "rollout/viol_RI2_rate",
            "viol_RG4": "rollout/viol_RG4_rate",
            "viol_RG2": "rollout/viol_RG2_rate"}
    out = pd.DataFrame()
    for k, c in cols.items():
        out[k] = df[c] if c in df.columns else np.nan
    return out.dropna(subset=["t"])


def plot_training_curves(cfg, tags=("ppo", "dqn", "qrdqn"), save=True):
    """Plot reward, length, overtakes, lane changes, crash rate — one figure each."""
    panels = [("rew", "Mean Episode Reward"), ("len", "Mean Episode Length"),
              ("overtakes", "Overtakes per Episode"),
              ("lane_changes", "Lane Changes per Episode"), ("crash", "Crash Rate")]
    curves = {tag: load_curve(cfg, tag) for tag in tags}
    figs = []
    for col, title in panels:
        fig, ax = plt.subplots(figsize=(10, 4))
        for tag in tags:
            df = curves[tag]
            if df is None or df.empty:
                continue
            d = df[["t", col]].dropna().sort_values("t")
            ax.plot(d["t"], d[col], marker=".", markersize=3, label=tag.upper())
        ax.set(title=f"Training: {title}", xlabel="timesteps", ylabel=col)
        if col == "crash":
            ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        if save:
            fig.savefig(drive_path(cfg, "metrics", f"training_{col}.png"),
                        dpi=120, bbox_inches="tight")
        figs.append(fig)
    return figs


def plot_eval_comparison(metrics_by_name, cfg=None, save_as=None):
    """Side-by-side bar chart comparing evaluation metrics across models."""
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


def plot_finetune_violations(cfg, tag="part2_nesy", save=True):
    """Plot per-rule violation rates during fine-tuning (RI2, RG4, RG2)."""
    df = load_curve(cfg, tag)
    if df is None or df.empty:
        return None
    rules = [("viol_RI2", "RI2: Passing on Right"),
             ("viol_RG4", "RG4: Impeding Flow"),
             ("viol_RG2", "RG2: Abrupt Braking")]
    fig, ax = plt.subplots(figsize=(10, 4))
    for col, label in rules:
        d = df[["t", col]].dropna().sort_values("t")
        if not d.empty:
            ax.plot(d["t"], d[col], marker=".", markersize=3, label=label)
    ax.set(title="Fine-tune: Soft-Rule Violation Rates (lower is better)",
           xlabel="timesteps", ylabel="violation rate")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    if save:
        fig.savefig(drive_path(cfg, "metrics", "finetune_violations.png"),
                    dpi=120, bbox_inches="tight")
    return fig


RULES = ("RG1", "RG2", "RG3", "RG4", "RI1", "RI2")


def plot_violation_rates(metrics_by_name, cfg=None, save_as="violation_rates.png"):
    """Grouped bar chart of per-rule violation rates across configs."""
    names = list(metrics_by_name)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(RULES))
    width = 0.8 / max(len(names), 1)
    for i, n in enumerate(names):
        rv = metrics_by_name[n]["summary"].get("rule_violation_rate", {})
        vals = [rv.get(r, 0.0) for r in RULES]
        ax.bar(x + i * width, vals, width, label=n)
    ax.set(title="Per-rule violation rate — lower is better",
           xlabel="rule", ylabel="fraction of steps violating")
    ax.set_xticks(x + width * (len(names) - 1) / 2)
    ax.set_xticklabels(RULES)
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    if cfg is not None and save_as:
        fig.savefig(drive_path(cfg, "metrics", save_as), dpi=120, bbox_inches="tight")
    return fig
