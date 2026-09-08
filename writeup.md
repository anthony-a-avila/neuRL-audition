# Recreating Schultz, Dayan & Montague (1997) with a TD learner

**Setup.** Trial of 80 time steps, cues at t=10 and t=20, reward r=1 at t=60, exactly as in Fig. 3 of the
paper. Each cue is a complete serial compound (one weight per delay), V(t) is a linear sum of the active
weights, delta(t) = r(t) + gamma*V(t+1) - V(t), and weights are updated once per trial by Eq. 5. Defaults are
alpha=0.1, gamma=1, 400 trials, 8 seeds. Randomness comes from reward magnitude (SD 0.15), 10% reward
omission, +/-2 steps of cue-to-reward jitter, near-zero weight init, and spike-count noise (SD 0.05) added to
the *reported* delta only, never to the learning signal.

## What I found

All three patterns come out (`figures/fig1_three_cases.png`, mean +/- SD over 8 seeds at the relevant time step):

| case | delta at cue (t=10) | delta at reward (t=60) |
| --- | --- | --- |
| naive, reward occurs | 0.00 | +0.91 +/- 0.13 |
| trained, reward occurs | +0.88 +/- 0.03 | +0.05 +/- 0.08 |
| trained, reward withheld | +0.88 +/- 0.03 | -0.94 +/- 0.07 |

Two further predictions from the paper also hold. The response transfers to the *earliest* cue only: after
training, delta at the second cue (t=20) is 0.02 +/- 0.03, indistinguishable from baseline. And the model
learns *when* the reward arrives long before it learns to respond to the cue - the reward-time response is
below 0.1 by trial 11, but the cue response does not reach 0.9 until trial 354. The paper notes this same
ordering for its own figure. The reason is visible in `figures/fig3_surfaces.png`: with a serial compound
there is no eligibility beyond one step, so value backs up exactly one time step per trial, and bridging the
50-step gap therefore costs at least 50 trials regardless of the learning rate.

## Where my version differs from the paper's

**gamma has to be 1.** The value surface in the paper's Fig. 3 saturates at 1, which is only possible with no
discounting. That is not a free choice: the cue-time response scales as gamma^50 over a 50-step interval, so
gamma=0.98 gives a cue response of 0.359 (gamma^50 = 0.364) and gamma=0.9 gives 0.005. A correctly
implemented TD model with any conventional discount factor produces a *much* smaller cue response than the
one the paper plots. This is worth flagging because the figure is often read as showing a full-size transfer.

**The omission dip is fragile in a way the figure hides.** It only reaches -1 when the cue-to-reward interval
is fixed. With +/-2 steps of jitter the dip collapses to -0.31 +/- 0.22, and to -0.18 +/- 0.19 with all five
noise sources on (`figures/noise_ablation.png`, panel B); jitter is also by far the largest source of
across-seed variability (panel A). The fixed point explains the size: V(t) converges to the probability that
the reward is still to come, so under uniform jitter over 5 steps the dip is V(61) - V(60) = 0.4 - 0.6 = -0.2,
which the measurements scatter around without settling on, because the jitter keeps moving the weights near
the reward. This is why the main figure holds timing fixed and treats jitter separately. It is also consistent
with the paper's own footnote 18 (Hollerman & Schultz), where depressions at the normal reward time appear
only when the reward is actually late.

**Speed.** The paper's Fig. 3 completes the transfer within the ~60 trials it plots; at alpha=0.1 mine needs
about 350. The surfaces figure uses alpha=0.5 to fit the transfer into a comparable number of trials.

## What I tried that did not work

- Letting the serial compound include a component at the cue's own onset (delay 0 rather than delay 1). This
  looks harmless and quietly destroys the result: the cue response falls to 0.008, because the weight at delay
  0 learns to predict the reward and there is then nothing preceding it to be surprised by. The onset spike
  exists only because the cue is unrepresented at the moment it appears.
- alpha above 0.5 diverges, but only with two cues. Both compounds are active at the same time steps, so the
  effective step size per time step is 2*alpha and stability needs 2*alpha < 1. With a single cue alpha=1 is
  stable. Nothing in Eq. 5 signals this.
- Feeding the spike-count noise into learning rather than only into the recorded trace. It is fairly benign
  (cue response 0.99 -> 0.81 at SD 0.2), but it conflates measurement noise with a learning signal, so the
  code keeps the two separate.
- A vectorized value lookup with `np.take` instead of `np.take_along_axis`, which silently flattens the weight
  matrix. It produced an output that looked superficially reasonable but put the trained peak at t=66 and made
  the omission response *positive*. Checking the sign the omission case must have is what caught it.

## What I would try next

Replace the serial compound with a microstimulus / temporal-basis representation (Ludvig, Sutton & Kehoe 2008)
and rerun the jitter ablation. The serial compound assumes a perfect internal clock, which is exactly the
assumption that makes the omission dip collapse under interval jitter. A coarse, overlapping basis should
smear the prediction over neighbouring time steps and may preserve a dip under jitter that the serial compound
cannot - and it would be a concrete test of whether the temporal representation, rather than the learning
rule, is what limits generalization to a shifted reward time.
