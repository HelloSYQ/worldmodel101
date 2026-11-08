# Step 0: given everything

> Ladder · Week 1 · Code: [`run.py`](run.py) · Shared: [`pendulum.py`](../pendulum.py), [`planner.py`](../planner.py), [`evaluate.py`](../evaluate.py)

**What you're given:** the true equations, the noise statistics and an angle sensor.
**What's new:** only the planner. This step sets the baseline every later step is compared against, and builds the three files they all reuse.

---

## 1. The world

A point mass on a rod, $\theta=0$ upright, $\theta=\pi$ hanging down. Semi-implicit Euler with $\Delta t = 0.05$ s:

```math
\begin{aligned}
\omega_{t+1} &= \omega_t + \Delta t\left(\frac{g}{l}\sin\theta_t - \frac{b}{ml^2}\,\omega_t + \frac{u_t + w_t}{ml^2}\right), \qquad w_t\sim\mathcal N(0,\sigma_w^2)\\
\theta_{t+1} &= \theta_t + \Delta t\,\omega_{t+1}\\
y_t &= \mathrm{wrap}(\theta_t + v_t), \qquad v_t\sim\mathcal N(0,\sigma_v^2)
\end{aligned}
```

| Quantity | Value | Why it matters |
|---|---|---|
| torque limit $u_{\max}$ | 2 N·m | gravity torque $mgl = 9.81$ N·m, so it can't lift directly and must pump about five times (about 7 s) |
| torque disturbance $\sigma_w$ | 0.1 N·m | process noise: even the true model can't predict exactly |
| angle sensor noise $\sigma_v$ | 0.05 rad | about 3°; no velocity sensor |
| cost | $\mathrm{wrap}(\theta)^2 + 0.1\,\omega^2 + 0.001\,u^2$ | reward $= -$cost; 300 steps (15 s) per episode |

## 2. The estimator: EKF

You know this one. Two details specific to this problem:

- **Process noise enters through the torque**, so $Q = \sigma_w^2\, G G^\top$ with $G = \partial x_{t+1}/\partial w_t = [\Delta t^2/ml^2,\; \Delta t/ml^2]^\top$. It's rank one: the disturbance pushes $\theta$ and $\omega$ in a fixed ratio.
- **The innovation is wrapped**: $\nu = \mathrm{wrap}(y - \hat\theta)$. Without this, crossing $\pm\pi$ gives a $2\pi$ innovation and the filter jumps.

Consistency check: the normalized innovation squared $\mathrm{NIS}_t = \nu_t^2/S_t$ averages 1 when the filter's model and noise statistics are right.

## 3. The planner: MPPI (new)

The planning problem is

```math
\min_{u_{0:H-1}} \; S(U) = \sum_{t=0}^{H-1} c(x_t, u_t) \quad \text{s.t.} \quad x_{t+1} = f(x_t, u_t),
```

with $H = 30$ steps (1.5 s). $S$ is non-convex, since pumping means pushing *away* from the goal first, so gradient methods get stuck. MPPI uses only samples instead.

Let $p(U)$ be a Gaussian around the current plan $\bar U$. Look for the distribution $q$ that keeps the expected cost low without moving far from $p$:

```math
q^\star = \arg\min_q \; \mathbb E_q[S(U)] + \lambda\, D_{\mathrm{KL}}(q \,\Vert\, p)
\quad\Longrightarrow\quad
q^\star(U) \propto p(U)\, e^{-S(U)/\lambda}.
```

Draw $U^{(k)} \sim p$, $k = 1..K$, and estimate the new plan $\mathbb E_{q^\star}[U]$ by importance sampling:

```math
\bar U \leftarrow \sum_k w_k\, U^{(k)}, \qquad w_k = \frac{e^{-S(U^{(k)})/\lambda}}{\sum_j e^{-S(U^{(j)})/\lambda}}.
```

The temperature $\lambda$ trades greed for averaging. As $\lambda \to 0$ you keep only the best sample; as $\lambda \to \infty$ you average all samples and ignore cost. Then apply $\bar u_0$, shift the plan one step and replan at the next measurement (receding horizon). Replanning closes the loop, which is why MPC tolerates some model error.

Simplifications here: the full MPPI cost has an extra control term from the sampling noise, and the planner treats the state estimate as exact. CEM, which you'll use in step 4, is the same loop with a hard top-$k$ selection and a refitted Gaussian instead of soft weights.

## 4. The evaluation (the same for every step)

- **Control:** episode return, and the percentage of the last 5 s spent upright ($|\theta| < 0.3$ rad). 5 episodes, env seeds 0–4.
- **Prediction:** open-loop $\theta$ RMSE after $k$ steps, from 500 random start states under random actions, against the true simulator *with* process noise. From step 1 this shows what the learned model got wrong, separately from the planner.

---

## 5. Predict before you run

Write your answers under **Your predictions** below (or in chat) **before** running `run.py`.

1. **Part A, true model.** Is its $k$-step error zero? If not, why not, and how should it grow with $k$? (Hint: what does the simulator have that the model's mean equations don't?)
2. **Part A, wrong model** ($l = 1.3$ m, 30% too long). Roughly how much bigger is its error at $k = 40$ than at $k = 1$?
3. **Part B.** Rank the four agents (`oracle`, `diff`, `ekf`, `ekf-wrong`) by return.
4. **Part B, `diff`.** From $\sigma_v$ and $\Delta t$, estimate the noise std of the finite-difference velocity $(y_t - y_{t-1})/\Delta t$. Will `diff` still swing up and balance?
5. **Part B, `ekf-wrong`.** Its open-loop model is badly off over 2 s, but it replans every 50 ms. Will feedback save it? Will its mean NIS be near 1 or far from it, and why?

### Your predictions

1.
2.
3.
4.
5.

## 6. Run

```bash
python3 -m ladder.step0.run --plot
```

About 10 s on a CPU, numpy only (`--plot` needs matplotlib and saves `outputs/ladder_step0.png`).

## 7. Results

Fill in after running, then copy the `oracle` and `ekf` rows to the [scoreboard](../README.md#scoreboard).

**Part A: $\theta$ RMSE [rad] after $k$ steps**

| Model | k=1 | k=5 | k=10 | k=20 | k=40 |
|---|---|---|---|---|---|
| true | | | | | |
| wrong ($l=1.3$) | | | | | |

**Part B: swing-up**

| Agent | Return | Upright % | ω estimate RMSE | Mean NIS |
|---|---|---|---|---|
| oracle | | | | — |
| diff | | | | — |
| ekf | | | | |
| ekf-wrong | | | | |

**What surprised me** (compare with your predictions):

## 8. Exercises (after comparing)

Each one is a one-line change in `run.py` or `planner.py`.

1. **Horizon.** Set the MPPI horizon to 20 and then 10 steps for `oracle`. What happens, and why can't a short horizon swing up? What would the cost need to include for a 10-step horizon to work? (Step 5's learned value function does this job.)
2. **Filter tuning.** Give the `ekf` agent $\sigma_w$ ten times too small, then ten times too large. Watch NIS and the ω RMSE. Which mistake hurts control more?
3. **Samples and temperature.** Cut MPPI to $K = 50$ samples. Then try $\lambda = 0.1$ and $\lambda = 10$ with $K = 500$. Explain what you see using the KL objective in §3.
4. **Hiding model error.** Keep $l = 1.3$ in `ekf-wrong`'s planner and filter, but inflate its filter's $\sigma_w$. Does control improve? What does this say about treating model error as process noise, and about the NIS alarm?
5. **Research thinking.** In step 1 the model is learned from random-action data, and Part A measures it under random actions too. The planner, though, drives the pendulum to states that random actions rarely reach (balanced upright, fast swings). Why might a model with good Part A numbers still fail at control?

## What carries forward

`pendulum.py`, `planner.py` and `evaluate.py` stay fixed. **Step 1 takes away the equations:** the planner's model `functools.partial(dynamics, p=TRUE)` becomes a neural network trained on transitions from random actions. The sensor switches to `state`, so the model is the only thing that changes.
