# TD model of dopamine prediction errors

A replication of Schultz, Dayan & Montague (1997), *A Neural Substrate of Prediction and Reward*, Science
275:1593. A TD learner with a complete serial compound representation is trained on a cue-then-reward trial;
its prediction error reproduces the three dopamine response patterns of the paper's Fig. 1 and the learning
surfaces of its Fig. 3.

Findings are in [writeup.md](writeup.md).

## Running it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python make_figures.py     # writes figures/*.png, takes a couple of seconds
```

`.venv/bin/python td_dopamine.py` prints a one-line summary of each of the three cases without plotting.

## Files

| file | contents |
| --- | --- |
| [td_dopamine.py](td_dopamine.py) | the model: serial compound representation, TD error, Eq. 5 update, noise sources |
| [make_figures.py](make_figures.py) | the three figures |
| `figures/fig1_three_cases.png` | naive / trained / reward-withheld responses over 8 seeds (Fig. 1) |
| `figures/fig3_surfaces.png` | prediction error and value as functions of trial and time (Fig. 3) |
| `figures/noise_ablation.png` | which source of randomness drives the across-seed spread |

Parameters live in `td_dopamine.Config`; `td_dopamine.NOISE_SOURCES` lists the sources of randomness and
`with_noise(cfg, *names)` switches them on individually.
