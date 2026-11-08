# Track L: the ladder (build-first)

A second way through the same material as the [main syllabus](../README.md#syllabus-and-progress). The main track teaches theory first and builds later. This track builds one world model in week 1, upgrades it one piece at a time, and brings in theory when something breaks.

Both tracks end in the same place. Progress on one doesn't depend on the other.

## The idea

Step 0 gives you everything: the true equations, the noise statistics and a sensor. Each later step **takes away one thing you were given** and replaces it with something learned. Each step ends with a working agent, and the way it fails shows what the next step has to fix.

| Step | Taken away | What you build | How it fails | Theory you learn here | Paper | Status |
|---|---|---|---|---|---|---|
| [0](step0/) | nothing | Pendulum simulator, EKF on a noisy angle sensor, MPPI planner on the true equations | — (the baseline) | MPPI; everything else you know | Williams et al. 2017 (MPPI) | 🟡 predictions pending |
| 1 | the equations | MLP dynamics learned from random rollouts, same planner | Long rollouts drift, and the planner exploits model errors | MLE → MSE, backprop, Adam (main 0.1, 0.2) | Nagabandi et al. 2018 | ⬜ |
| 2 | trust in the model | Ensemble with learned variance | It still needs the true state | Epistemic vs aleatoric uncertainty | PETS (Chua et al. 2018) | ⬜ |
| 3 | the state | 32×32 rendered frames, VAE plus latent dynamics | One frame shows no velocity, and there's no filter now | VAE / ELBO (main 0.3) | Ha & Schmidhuber 2018 | ⬜ |
| 4 | the hand-built filter | RSSM, with CEM planning in latent space | Replanning every step is slow and short-sighted | Sequential ELBO, GRU (main 0.4) | PlaNet (Hafner et al. 2019) | ⬜ |
| 5 | the online planner | Actor-critic trained in imagination | — | RL, λ-returns (main 0.5, 0.6) | DreamerV1, then the V3 tricks | ⬜ |
| 6 | reconstruction | Pick one: decoder-free (TD-MPC2), tokens + transformer (IRIS), diffusion (DIAMOND), or JEPA | | Main Phases 4–7 | | ⬜ |
| 7 | the pendulum | The same stack on CDL channel sequences, with beam choice as the action | | Main Track W | TR 38.843 | ⬜ |

Pace: about one step per week at 5 h/week. Steps 4 and 5 may take two weeks each.

## Why this suits an estimation background

The RSSM in step 4 is a learned Kalman filter. By then you'll have written both, so you can check this mapping in your own code:

| Kalman filter (step 0) | RSSM (step 4) |
|---|---|
| state $x_t$ | latent $(h_t, z_t)$ |
| transition $F, B$ | GRU + prior $p(z_t \mid h_t)$ |
| observation $H$ | decoder $p(o_t \mid h_t, z_t)$ |
| predict step | prior |
| update step | posterior $q(z_t \mid h_t, o_t)$ |
| innovation size | $\mathrm{KL}(q \,\Vert\, p)$: how far the measurement moved the belief |
| system ID via EM | training on the ELBO |
| MPC | CEM in latent space (step 4), or a policy trained in imagination (step 5) |

## Rules

- **You own the simulator.** You know the true dynamics, so every step reports open-loop prediction error against ground truth, not only task return.
- **Same planner, same evaluation, same seeds at every step.** [`planner.py`](planner.py) and [`evaluate.py`](evaluate.py) don't change, so each step is a measured difference from the one before.
- **Predict before you run.** Each step's README asks for predictions first. The gap between what you expected and what happened is the lesson.
- **Code first, notes after.** Notes record what broke and why the fix works. Read each paper when you hit the problem it solves.
- **From scratch.** No world-model libraries (numpy, then PyTorch), small enough to run on a CPU. From step 1, try writing the new piece from the brief before looking at a reference.

## Layout

```
ladder/
  pendulum.py    the world: true equations, sensors, cost (never changes)
  planner.py     MPPI, reused by every step that plans
  evaluate.py    the shared metrics: control and k-step prediction error
  stepN/
    README.md    brief, predictions, results, exercises
    run.py       the experiment
```

Run from the repo root, for example `python3 -m ladder.step0.run`. Plots go to `outputs/`, which git ignores.

## Scoreboard

One row per agent, filled in after running each step. All rows use the same seeds.

| Step | Agent | Return | Upright % (last 5 s) | θ RMSE at k=40 [rad] |
|---|---|---|---|---|
| 0 | oracle (true model, true state) | | | |
| 0 | ekf (true model, angle sensor) | | | |
