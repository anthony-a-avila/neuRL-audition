# TD model of dopamine prediction errors

A replication of Schultz, Dayan & Montague (1997), *A Neural Substrate of Prediction and Reward*, Science
275:1593. A TD learner with a complete serial compound representation is trained on a cue-then-reward trial;
its prediction error reproduces the three dopamine response patterns of the paper's Fig. 1.

Everything lives in one notebook, [agent-sandbox.ipynb](agent-sandbox.ipynb), which derives the model alongside
the code and is committed with its outputs, so it reads without being run. The findings, the departures from
the paper and what I would try next are in [writeup.md](writeup.md).

## Running it

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/jupyter lab agent-sandbox.ipynb
```

Running all cells takes a couple of seconds and rewrites the two PNGs in `figures/`:

| figure | contents |
| --- | --- |
| `fig1_three_cases.png` | naive / trained / reward-withheld responses over 8 seeds (paper's Fig. 1) |
| `noise_ablation.png` | which source of randomness drives the across-seed spread |

To regenerate them without opening the notebook:

```sh
.venv/bin/jupyter nbconvert --to notebook --execute --inplace agent-sandbox.ipynb
```

Parameters live in the `Config` dataclass; `NOISE_SOURCES` lists the sources of randomness and
`with_noise(cfg, *names)` switches them on individually.
