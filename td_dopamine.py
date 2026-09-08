"""TD model of dopamine prediction-error signals.

Reproduces the model of Schultz, Dayan & Montague (1997), "A Neural Substrate of
Prediction and Reward", Science 275:1593.  Each sensory cue is a complete serial
compound: component tau of cue i is active exactly tau steps after that cue's
onset and carries its own weight, so

    V(t)     = sum_i w[i, t - onset_i]                          (Eq. 4)
    delta(t) = r(t) + gamma * V(t + 1) - V(t)                   (Eq. 3)
    dw[i, .] = alpha * sum_t x_i(t) delta(t)                    (Eq. 5)

Components start at tau = 1, as in the paper: nothing represents the cue at its
own onset, which is what leaves the phasic response at cue onset after learning.
"""

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Config:
    """Task and model parameters.  Defaults follow Fig. 3 of the paper."""

    n_steps: int = 80
    cue_times: tuple = (10, 20)
    reward_time: int = 60
    reward_size: float = 1.0
    alpha: float = 0.1
    gamma: float = 1.0
    n_trials: int = 400

    # Sources of across-seed variability, all off by default.
    reward_sd: float = 0.0  # trial-to-trial jitter in reward magnitude
    omission_prob: float = 0.0  # fraction of training trials with no reward
    timing_jitter: int = 0  # cue-to-reward interval varies by +/- this many steps
    weight_init_sd: float = 0.0  # weights start near zero instead of at zero
    report_noise_sd: float = 0.0  # spike-count noise on the *reported* delta only

    forced_omission_trials: tuple = ()  # trials where reward is withheld regardless


NOISE_SOURCES = {
    "reward magnitude": dict(reward_sd=0.15),
    "omission (10%)": dict(omission_prob=0.1),
    "timing jitter": dict(timing_jitter=2),
    "weight init": dict(weight_init_sd=0.01),
    "report noise": dict(report_noise_sd=0.05),
}


# Timing jitter is left out of the main three-case figure: unlike the others it
# changes the task itself, so the model can no longer place the reward exactly.
# Its effect is the subject of the ablation figure.
STOCHASTIC_REWARD = ("reward magnitude", "omission (10%)", "weight init", "report noise")


def with_noise(cfg=Config(), *names):
    """Return a copy of cfg with the named noise sources switched on."""
    settings = {}
    for name in names or NOISE_SOURCES:
        settings.update(NOISE_SOURCES[name])
    return replace(cfg, **settings)


def _delays(cfg):
    """delays[i, t] is the component of cue i active at time t, or -1 if none."""
    t = np.arange(cfg.n_steps)
    delays = np.stack([t - onset for onset in cfg.cue_times])
    return np.where((delays >= 1) & (delays < cfg.n_steps), delays, -1)


def trial(w, cfg, delays, reward_time, reward_size):
    """Run one trial at fixed weights; return V(t), delta(t) and the Eq. 5 update."""
    r = np.zeros(cfg.n_steps)
    if reward_time is not None:
        r[reward_time] = reward_size

    active = delays >= 0
    v = np.zeros(cfg.n_steps + 1)  # v[n_steps] = 0: the trial ends, nothing follows
    per_cue = np.take_along_axis(w, np.maximum(delays, 0), axis=1)
    v[: cfg.n_steps] = np.where(active, per_cue, 0.0).sum(axis=0)

    delta = r + cfg.gamma * v[1:] - v[: cfg.n_steps]

    dw = np.zeros_like(w)
    for i in range(len(cfg.cue_times)):
        on = active[i]
        dw[i, delays[i, on]] = cfg.alpha * delta[on]
    return v[: cfg.n_steps], delta, dw


@dataclass
class Result:
    deltas: np.ndarray  # (n_trials, n_steps) prediction error over training
    values: np.ndarray  # (n_trials, n_steps) value function over training
    weights: np.ndarray  # (n_cues, n_steps) weights after training
    omitted: np.ndarray  # (n_trials,) whether reward was withheld
    naive: np.ndarray  # delta(t) on the first trial
    trained: np.ndarray  # delta(t) after training, reward delivered
    omission: np.ndarray  # delta(t) after training, reward withheld


def run(cfg, seed):
    """Train on cfg for cfg.n_trials, then probe the trained model twice."""
    rng = np.random.default_rng(seed)
    delays = _delays(cfg)
    shape = (len(cfg.cue_times), cfg.n_steps)
    w = rng.normal(0.0, cfg.weight_init_sd, shape) if cfg.weight_init_sd else np.zeros(shape)

    deltas = np.zeros((cfg.n_trials, cfg.n_steps))
    values = np.zeros((cfg.n_trials, cfg.n_steps))
    omitted = np.zeros(cfg.n_trials, dtype=bool)

    for k in range(cfg.n_trials):
        # Trial 0 is always rewarded so that the naive-case trace is well defined.
        omit = k in cfg.forced_omission_trials or (rng.random() < cfg.omission_prob and k > 0)
        jitter = rng.integers(-cfg.timing_jitter, cfg.timing_jitter + 1)
        reward_time = None if omit else cfg.reward_time + jitter
        reward_size = cfg.reward_size + cfg.reward_sd * rng.standard_normal()

        v, delta, dw = trial(w, cfg, delays, reward_time, reward_size)
        omitted[k] = omit
        values[k] = v
        deltas[k] = delta + cfg.report_noise_sd * rng.standard_normal(cfg.n_steps)
        w += dw  # learning uses delta itself; the recorded trace is the noisy one

    def probe(omit):
        _, delta, _ = trial(w, cfg, delays, None if omit else cfg.reward_time, cfg.reward_size)
        return delta + cfg.report_noise_sd * rng.standard_normal(cfg.n_steps)

    return Result(deltas, values, w, omitted, deltas[0], probe(omit=False), probe(omit=True))


if __name__ == "__main__":
    cfg = Config()
    res = run(cfg, seed=0)
    print(f"cues at {cfg.cue_times}, reward at {cfg.reward_time}, {cfg.n_trials} trials")
    for name, delta in (("naive", res.naive), ("trained", res.trained), ("omission", res.omission)):
        print(f"{name:9s} peak |delta| = {np.abs(delta).max():+.3f} at t = {np.abs(delta).argmax()}")
