"""Generate the figures for the TD / dopamine replication.

Writes three PNGs into figures/:
  fig1_three_cases.png  the three response patterns of Fig. 1, over several seeds
  fig3_surfaces.png     the prediction error and value surfaces of Fig. 3
  noise_ablation.png    which source of randomness drives the across-seed spread
"""

from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import td_dopamine as td

SEEDS = range(8)
FIGURES = Path(__file__).parent / "figures"


def mark_events(ax, cfg, reward_label="R"):
    for onset in cfg.cue_times:
        ax.axvline(onset, color="tab:blue", lw=1, ls="--", zorder=0)
    ax.axvline(cfg.reward_time, color="tab:red", lw=1, ls=":", zorder=0)
    ax.annotate("CS", (cfg.cue_times[0], 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", color="tab:blue", fontsize=9)
    ax.annotate(reward_label, (cfg.reward_time, 0.97), xycoords=("data", "axes fraction"),
                ha="center", va="top", color="tab:red", fontsize=9)


def fig_three_cases(cfg):
    """Fig. 1: naive, trained, and reward-omitted responses across seeds."""
    runs = [td.run(cfg, seed) for seed in SEEDS]
    panels = [
        ("A  No prediction, reward occurs (trial 1)", [r.naive for r in runs], "R"),
        ("B  Reward predicted, reward occurs (trained)", [r.trained for r in runs], "R"),
        ("C  Reward predicted, no reward occurs (trained)", [r.omission for r in runs], "(no R)"),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 8.0), sharex=True, sharey=True)
    for ax, (title, traces, reward_label) in zip(axes, panels):
        traces = np.array(traces)
        mark_events(ax, cfg, reward_label)
        ax.axhline(0, color="0.6", lw=0.8, zorder=0)
        for trace in traces:
            ax.plot(trace, color="0.55", lw=0.8, alpha=0.7, zorder=1)
        ax.plot(traces.mean(axis=0), color="k", lw=1.8, zorder=2,
                label=f"mean of {len(SEEDS)} seeds")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_ylabel(r"TD error  $\delta(t)$")
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    axes[-1].set_xlabel("time step within trial")
    axes[0].set_xlim(0, cfg.n_steps - 1)
    axes[0].set_ylim(-1.35, 1.45)
    fig.suptitle("TD prediction error reproduces the three dopamine response patterns\n"
                 f"cues at t={cfg.cue_times}, reward at t={cfg.reward_time}, "
                 rf"$\alpha$={cfg.alpha}, $\gamma$={cfg.gamma}, {cfg.n_trials} training trials",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def fig_surfaces(cfg):
    """Fig. 3: prediction error and value as functions of trial and time."""
    res = td.run(cfg, seed=0)
    trial, time = np.meshgrid(np.arange(cfg.n_trials), np.arange(cfg.n_steps), indexing="ij")

    fig = plt.figure(figsize=(11.0, 5.0))
    for i, (surface, label) in enumerate(((res.deltas, r"Prediction error  $\delta(t)$"),
                                          (res.values, r"Value  $V(t)$"))):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        ax.plot_surface(time, trial, surface, cmap="viridis", rstride=1, cstride=1,
                        linewidth=0, antialiased=True)
        ax.set_xlabel("time step within trial", labelpad=8)
        ax.set_ylabel("trial", labelpad=6)
        ax.set_title(label, fontsize=10)
        ax.set_zlim(-1.1, 1.1)
        ax.view_init(elev=42, azim=-72)

    omitted = ", ".join(str(k) for k in cfg.forced_omission_trials)
    fig.suptitle("Development of the prediction error through training "
                 f"(reward withheld on trial {omitted})", fontsize=11, y=0.98)
    fig.subplots_adjust(left=0.0, right=0.99, bottom=0.06, top=0.90, wspace=0.05)
    return fig


def fig_noise_ablation(cfg):
    """Across-seed spread of the trained responses, one noise source at a time."""
    conditions = {"none": (), **{name: (name,) for name in td.NOISE_SOURCES},
                  "all": tuple(td.NOISE_SOURCES)}

    readouts = {}
    for label, names in conditions.items():
        noisy = td.with_noise(cfg, *names) if names else cfg
        runs = [td.run(noisy, seed) for seed in SEEDS]
        readouts[label] = np.array([
            [r.trained[cfg.cue_times[0]], r.trained[cfg.reward_time], r.omission[cfg.reward_time]]
            for r in runs
        ])

    labels = list(conditions)
    x = np.arange(len(labels))
    names = ("cue response", "reward response", "omission dip")

    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9.0, 7.0))

    width = 0.26
    for i, name in enumerate(names):
        spread = [readouts[label][:, i].std() for label in labels]
        top.bar(x + (i - 1) * width, spread, width, label=name)
    top.set_xticks(x, labels, rotation=20, ha="right")
    top.set_ylabel(r"across-seed SD of $\delta$")
    top.set_title("A  Which noise source drives the spread across seeds", loc="left", fontsize=10)
    top.legend(fontsize=8, frameon=False)

    dip = np.array([readouts[label][:, 2] for label in labels])
    bottom.errorbar(x, dip.mean(axis=1), yerr=dip.std(axis=1), fmt="o", capsize=4, color="k")
    bottom.axhline(-cfg.reward_size, color="tab:red", ls=":", lw=1,
                   label="full dip expected without noise")
    bottom.set_xticks(x, labels, rotation=20, ha="right")
    bottom.set_ylabel(r"$\delta$ at reward time, reward withheld")
    bottom.set_title("B  Timing jitter does not just add variance, it shrinks the dip",
                     loc="left", fontsize=10)
    bottom.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Effect of each source of randomness ({len(SEEDS)} seeds each)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def main():
    FIGURES.mkdir(exist_ok=True)
    base = td.Config()

    figures = {
        "fig1_three_cases.png": fig_three_cases(td.with_noise(base, *td.STOCHASTIC_REWARD)),
        # A faster learning rate keeps the transfer to the cue visible within the
        # number of trials the paper plots.
        "fig3_surfaces.png": fig_surfaces(
            replace(base, alpha=0.5, n_trials=120, forced_omission_trials=(100,))
        ),
        "noise_ablation.png": fig_noise_ablation(base),
    }
    for name, fig in figures.items():
        fig.savefig(FIGURES / name, dpi=200)
        plt.close(fig)
        print(f"wrote {FIGURES / name}")


if __name__ == "__main__":
    main()
