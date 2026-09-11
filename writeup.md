# Recreating Schultz, Dayan & Montague (1997) with a TD learner

**Setup.** Trial of 80 time steps, cue at t=10, reward r=1 at t=60, for the Fig. 1 replication task. The cue is a complete serial compound (one weight per delay), V(t) is a linear sum of the active
weights, delta(t) = r(t) + gamma*V(t+1) - V(t), and weights are updated once per trial by Eq. 5. Defaults are
alpha=0.1, gamma=1, 650 training trials (one cue needs more trials than the paper's two-cue task for the same cue-time response), 8 seeds. Randomness comes from reward magnitude (SD 0.15), 10% reward
omission, +/-2 steps of cue-to-reward jitter, near-zero weight init, and spike-count noise (SD 0.05) added to
the *reported* delta only, never to the learning signal.

## What I found

All three patterns come out (`figures/fig1_three_cases.png`, mean +/- SD over 8 seeds at the relevant time step):

| case | delta at cue (t=10) | delta at reward (t=60) |
| --- | --- | --- |
| naive, reward occurs | 0.00 | +0.97 +/- 0.12 |
| trained, reward occurs | +0.90 +/- 0.05 | +0.10 +/- 0.10 |
| trained, reward withheld | +0.90 +/- 0.05 | -0.89 +/- 0.09 |

Two further predictions from the paper also hold. The model learns *when* the reward arrives long before it learns to respond to the cue - the reward-time response is
below 0.1 by trial 11, but the cue response does not reach 0.9 until roughly trial 600 with a single serial compound at this learning rate. The paper notes this same
ordering for its own figure. The reason is that a serial compound has no eligibility beyond one step, so value
backs up exactly one time step per trial: V(t) first becomes nonzero on trial 61-t for every t, at any
learning rate. Bridging the 50-step gap therefore costs at least 50 trials no matter how alpha is set, while
cancelling the reward-time response only takes a handful.

## Where my version differs from the paper's

**gamma has to be 1.** The value surface in the paper saturates at 1, which is only possible with no
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

**Speed.** The paper completes the transfer within the ~60 trials it plots; with one cue at alpha=0.1 the
cue response needs on the order of 600 training trials to exceed 0.9. At alpha=0.5 a single cue reaches full
transfer within 400 trials, and alpha=1 remains stable with only one compound.

## What I tried that did not work

- Letting the serial compound include a component at the cue's own onset (delay 0 rather than delay 1). This
  looks harmless and quietly destroys the result: the cue response falls to 0.008, because the weight at delay
  0 learns to predict the reward and there is then nothing preceding it to be surprised by. The onset spike
  exists only because the cue is unrepresented at the moment it appears.
- With multiple simultaneous cues, stability needs n_cues * alpha < 1 because overlapping compounds update the
  same time steps more than once per trial. This replication uses a single cue, so alpha=1 is stable; nothing
  in Eq. 5 signals the multi-cue limit.
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
