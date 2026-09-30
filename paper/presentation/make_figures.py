"""Draw every chart of the presentation (and README) from the real results in data/.

Run:  python make_figures.py        -> figures/*.pdf (slides) + figures/*.png (README)

data/ holds copies of the notebooks' metrics (Drive: nesy-highway-driving/metrics/):
the Part-1 eval JSONs + training curves, the Part-2 and Part-3 eval JSONs, and two
replays of the same test runs (action_mix.json, shield_causes.json).
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
OUT = os.path.join(HERE, "figures")

# ── style: one look for every chart ───────────────────────────────────────────────────────────
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#ffffff"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#b9b7af"
AGENTS = [("dqn", "DQN", BLUE), ("qrdqn", "QR-DQN", ORANGE), ("ppo", "PPO", AQUA)]
HARD, SOFT = ["RG1", "RG3", "RI1"], ["RG2", "RG4", "RI2"]
RULE_NAME = {"RG1": "safe distance", "RG3": "speed limit", "RI1": "no stopping",
             "RG2": "no harsh braking", "RG4": "keep the flow", "RI2": "no right pass"}
SEQ = LinearSegmentedColormap.from_list("seq", ["#f4f8fd", "#b7d3f6", "#5598e7", "#1c5cab", "#0d366b"])


def use_lato():
    """Use Lato (the slides' font) when the TeX tree has it; fall back to the default sans."""
    try:
        import subprocess
        path = subprocess.run(["kpsewhich", "Lato-Regular.ttf"], capture_output=True, text=True).stdout.strip()
    except OSError:
        path = ""
    if path:
        for w in ("Regular", "Bold", "Semibold"):
            p = path.replace("Regular", w)
            if os.path.exists(p):
                font_manager.fontManager.addfont(p)
        plt.rcParams["font.family"] = "Lato"


def style():
    use_lato()
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.titlepad": 6, "axes.edgecolor": AXIS, "axes.linewidth": 0.6, "axes.labelcolor": INK2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "text.color": INK, "legend.frameon": False,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "pdf.fonttype": 42,
    })


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def load(name):
    with open(os.path.join(DATA, f"{name}.json")) as f:
        return json.load(f)


def bar_end_label(ax, bars, fmt, pad):
    for b in bars:
        w = b.get_width()
        ax.text(w + pad, b.get_y() + b.get_height() / 2, fmt(w), va="center", ha="left", color=INK, fontsize=7)


# ── Part 1 ────────────────────────────────────────────────────────────────────────────────────
def p1_training():
    panels = [("rollout/ep_crash_rate", "Crash rate", 100, "%"),
              ("rollout/ep_overtakes_mean", "Overtakes per run", 1, ""),
              ("rollout/ep_len_mean", "Run length (steps)", 1, "")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.95))
    for ax, (col, title, scale, unit) in zip(axes, panels):
        for tag, name, color in AGENTS:
            df = pd.read_csv(os.path.join(DATA, f"curves_{tag}.csv"))
            d = df[["time/total_timesteps", col]].dropna().sort_values("time/total_timesteps")
            x, y = d["time/total_timesteps"] / 1000, d[col] * scale
            ax.plot(x, y, color=color, lw=1.5, solid_capstyle="round", label=name)
            ax.plot(x.iloc[-1], y.iloc[-1], "o", ms=4, color=color, mec=SURFACE, mew=1.0)
        ax.set_title(title)
        ax.set_xlabel("training steps (thousands)")
        ax.set_xlim(0, 20.5)
        if unit == "%":
            ax.set_ylim(0, 105)
            ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    axes[0].legend(loc="lower left", fontsize=7, handlelength=1.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "p1_training")


def p1_final_test():
    panels = [("crash_rate", "Crash rate", lambda s: 100 * s["crash_rate"], lambda v: f"{v:.0f}%"),
              ("overtakes", "Overtakes per run", lambda s: s["overtakes"]["mean"], lambda v: f"{v:.1f}"),
              ("distance", "Distance per run", lambda s: s["distance"]["mean"], lambda v: f"{v:.0f} m")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.3), sharey=True)
    names = [n for _, n, _ in AGENTS]
    for ax, (_, title, get, fmt) in zip(axes, panels):
        vals = [get(load(tag)["summary"]) for tag, _, _ in AGENTS]
        bars = ax.barh(names, vals, height=0.55, color=[c for _, _, c in AGENTS])
        bar_end_label(ax, bars, fmt, max(vals) * 0.02)
        ax.set_xlim(0, max(vals) * 1.28)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.2)
    save(fig, "p1_final_test")


def p1_action_mix():
    mix = load("action_mix")
    order = ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"]
    labels = ["lane left", "keep (IDLE)", "lane right", "faster", "slower"]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 1.55), sharey=True)
    for ax, (tag, name, color) in zip(axes, AGENTS):
        counts = mix[tag]["actions"]
        total = sum(counts.values())
        vals = [100 * counts.get(a, 0) / total for a in order]
        bars = ax.barh(labels, vals, height=0.55, color=color)
        bar_end_label(ax, bars, lambda v: f"{v:.0f}%", 1.5)
        ax.set_xlim(0, 118)
        ax.set_title(name)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    axes[0].invert_yaxis()
    fig.tight_layout(w_pad=1.2)
    save(fig, "p1_action_mix")


# ── rule heatmaps (Parts 2 and 3) ─────────────────────────────────────────────────────────────
def rule_heatmap(rows, name, best=None, artifact=None, height=2.0):
    """rows: [(label, json name)]. Columns: hard rules | soft rules, value = share of steps."""
    rules = HARD + SOFT
    m = np.array([[load(f)["summary"]["rule_violation_rate"][r] for r in rules] for _, f in rows])
    fig, ax = plt.subplots(figsize=(5.6, height))
    xs = [0, 1, 2, 3.25, 4.25, 5.25]  # a gap between hard and soft
    for i, row in enumerate(m):
        for j, v in enumerate(row):
            is_art = artifact and rules[j] == artifact
            face = "#eeede8" if is_art else SEQ(min(v / 0.8, 1.0))
            ax.add_patch(plt.Rectangle((xs[j] + 0.04, i + 0.06), 0.92, 0.88, color=face, lw=0))
            light = (not is_art) and v > 0.35
            txt = "0%" if v == 0 else ("<1%" if v < 0.005 else f"{100 * v:.0f}%")
            ax.text(xs[j] + 0.5, i + 0.5, txt,
                    ha="center", va="center", fontsize=7.5, color="white" if light else INK,
                    fontweight="bold" if light else "normal")
    ax.set_xlim(-0.05, 6.3)
    ax.set_ylim(len(rows), -0.45)
    ax.set_xticks([x + 0.5 for x in xs])
    ax.set_xticklabels([f"{r}\n{RULE_NAME[r]}" + ("\n(threshold artifact)" if r == artifact else "")
                        for r in rules], fontsize=7)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels([lab for lab, _ in rows])
    for t, (lab, _) in zip(ax.get_yticklabels(), rows):
        if lab == best:
            t.set_fontweight("bold")
            t.set_color(ORANGE)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(1.5, -0.2, "HARD rules  ·  blocked by the shield", ha="center", fontsize=7, color="#c62828", fontweight="bold")
    ax.text(4.75, -0.2, "SOFT rules  ·  penalised in the reward", ha="center", fontsize=7, color="#2e7d32", fontweight="bold")
    save(fig, name)


def p2_violations():
    rows = [("baseline (QR-DQN)", "p2_baseline"), ("+ shield", "p2_shield"), ("+ logic-reward", "p2_reward"),
            ("+ shield + reward", "p2_shield_reward"), ("+ MCTS", "p2_mcts"), ("+ MCTS + shield", "p2_mcts_shield")]
    rule_heatmap(rows, "p2_rule_violations", best="+ shield + reward", height=1.8)


def p3_violations():
    rows = [("brain only", "p3_bare"), ("+ shield", "p3_shield"), ("MCTS + shield", "p3_mcts")]
    rule_heatmap(rows, "p3_rule_violations", best="MCTS + shield", artifact="RI1", height=1.3)


def p2_tradeoff():
    rows = [("baseline", "p2_baseline"), ("+ shield", "p2_shield"), ("+ logic-reward", "p2_reward"),
            ("+ shield + reward", "p2_shield_reward"), ("+ MCTS", "p2_mcts"), ("+ MCTS + shield", "p2_mcts_shield")]
    pts = []
    for lab, f in rows:
        s = load(f)["summary"]
        pts.append((lab, 100 * s["crash_rate"], s["overtakes"]["mean"], sum(s["rule_violation_rate"].values())))
    base_crash, base_ot = pts[0][1], pts[0][2]
    fig, ax = plt.subplots(figsize=(3.7, 2.45))
    ax.add_patch(plt.Rectangle((0, base_ot / 2), base_crash, 7 - base_ot / 2, color="#eaf2fc", lw=0, zorder=0))
    ax.text(2, 6.8, "allowed: crashes ≤ baseline\nand ≥ half its overtakes", fontsize=6.5, color=BLUE, va="top")
    offsets = {"baseline": (-10, -3), "+ shield": (0, 10), "+ logic-reward": (0, -18), "+ shield + reward": (10, -6),
               "+ MCTS": (-10, 9), "+ MCTS + shield": (10, 2)}
    for lab, crash, ot, tot in pts:
        best = lab == "+ shield + reward"
        allowed = crash <= base_crash + 1e-9 and ot >= base_ot / 2
        ax.scatter(crash, ot, s=40 if best else 26, zorder=3, color=ORANGE if best else (INK2 if allowed else SURFACE),
                   edgecolors=ORANGE if best else INK2, linewidths=1.1)
        dx, dy = offsets[lab]
        ax.annotate(f"{lab} · {tot:.2f}", (crash, ot), xytext=(dx, dy), textcoords="offset points",
                    ha="right" if dx < 0 else ("left" if dx > 0 else "center"), fontsize=6.5, color=INK,
                    fontweight="bold" if best else "normal")
    ax.text(104, 0.25, "○ = not allowed", ha="right", fontsize=6.5, color=MUTED)
    ax.set_xlim(0, 105)
    ax.set_ylim(0, 7)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("crash rate  (lower is safer)")
    ax.set_ylabel("overtakes per run")
    ax.set_title("labels: total violations", fontsize=6.5, fontweight="normal", color=MUTED, loc="right")
    save(fig, "p2_tradeoff")


# ── why hard rules survive the shield ────────────────────────────────────────────────────────
def hard_rules_shield():
    series = [("highway, no shield", load("p2_baseline"), GREY),
              ("highway, + shield", load("p2_shield"), BLUE),
              ("MetaDrive, + shield + CBF", load("p3_shield"), AQUA)]
    rules = ["RG1", "RG3"]
    fig, ax = plt.subplots(figsize=(3.0, 1.75))
    y = np.arange(len(rules))
    h = 0.25
    for k, (lab, d, color) in enumerate(series):
        vals = [100 * d["summary"]["rule_violation_rate"][r] for r in rules]
        bars = ax.barh(y + (k - 1) * (h + 0.03), vals, height=h, color=color, label=lab)
        bar_end_label(ax, bars, lambda v: f"{v:.1f}%", 1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r}  {RULE_NAME[r]}" for r in rules])
    ax.invert_yaxis()
    ax.set_xlim(0, 75)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
    ax.set_xlabel("share of steps that break the rule")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6, handlelength=1.0, columnspacing=1.0)
    save(fig, "hard_rules_shield")


def shield_causes():
    """Replay of the '+shield' test runs: why each RG1 / RG3 violating step happened."""
    if not os.path.exists(os.path.join(DATA, "shield_causes.json")):
        return
    d = load("shield_causes")
    groups = {
        "rg1_cause": [("too close, braking takes time", ["already close, braking (lag)"]),
                      ("gap closed during the step", ["gap closed during the step (FASTER)",
                                                      "gap closed during the step (IDLE)",
                                                      "gap closed during the step (SLOWER)"]),
                      ("our own lane change", ["own lane change"]),
                      ("IDLE still allowed when too close", ["already close, shield allowed IDLE"]),
                      ("another car cut in", ["another car cut in"])],
        "rg3_cause": [("FASTER allowed just under the limit", ["FASTER allowed just under the limit"]),
                      ("over the limit, braking takes time", ["already over, braking (lag)"]),
                      ("lane change / IDLE still allowed", ["already over, shield allowed LANE_LEFT",
                                                           "already over, shield allowed LANE_RIGHT",
                                                           "already over, shield allowed IDLE"])],
    }
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 1.5))
    for ax, key, title in [(axes[0], "rg1_cause", f"RG1 safe distance ({d['violations']['RG1']} steps)"),
                           (axes[1], "rg3_cause", f"RG3 speed limit ({d['violations']['RG3']} steps)")]:
        total = sum(d[key].values())
        labels = [lab for lab, _ in groups[key]]
        vals = [100 * sum(d[key].get(k, 0) for k in keys) / total for _, keys in groups[key]]
        bars = ax.barh(labels, vals, height=0.6, color=BLUE)
        bar_end_label(ax, bars, lambda v: f"{v:.0f}%", 1.5)
        ax.set_xlim(0, 80)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
    fig.tight_layout(w_pad=1.5)
    save(fig, "shield_causes")


if __name__ == "__main__":
    style()
    os.makedirs(OUT, exist_ok=True)
    p1_training()
    p1_final_test()
    if os.path.exists(os.path.join(DATA, "action_mix.json")):
        p1_action_mix()
    p2_violations()
    p2_tradeoff()
    p3_violations()
    hard_rules_shield()
    shield_causes()
    print("figures written to", OUT)
