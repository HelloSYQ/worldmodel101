# Design: <name>

- **Author / date:**
- **Status:** idea / toy-tested / evaluated / capstone / abandoned
- **One-sentence pitch:** <what it predicts, from what data, used for what, and why that beats what exists>

> Copy this file to `designs/<short-name>.md`. Keep the first version to one page; sections 6–8 grow as you run experiments. A design is useful once section 5 says what result would prove it wrong.

## 1. The problem

- What fails in existing world models? Point to the specific evidence: paper, figure or table, and the setting it was measured in.
- Which open problem from Phase 10 does it address (memory, causality, uncertainty, hierarchy, scaling, real-time cost, learned vs. explicit physics, …)?
- Why does the failure happen? Name the mechanism, not just the symptom (e.g. "MSE on pixels averages the modes", not "the predictions are blurry").

## 2. Placement on the course grid

- **Predicts:** raw observations / latents with reconstruction / latents without reconstruction / task quantities only
- **Used for:** planning / policy in imagination / simulator / representations / data generation or evaluation
- **Nearest existing designs (2–3):** what each does, and the one thing this design does differently

## 3. The model

- **Variables:** observation $o_t$, latent state $s_t$, action $a_t$, reward $r_t$, anything else
- **Factorization:**

```math
% generative model, e.g.  p_\theta(o_{1:T}, s_{1:T} \mid a_{1:T}) = \prod_t p_\theta(s_t \mid s_{t-1}, a_{t-1})\, p_\theta(o_t \mid s_t)
% inference model, if any, e.g.  q_\phi(s_t \mid s_{t-1}, a_{t-1}, o_t)
```

- **Distribution family for each factor, and why:** fixed-variance Gaussian (MSE) / learned-variance Gaussian / categorical / mixture / diffusion or flow. Lesson 0.1 §7: a point estimate blurs a multi-modal future.
- **Training objective:** derive it from the factorization (likelihood, ELBO, or a stated surrogate), don't just write it down.
- **Assumptions:** Markov state? stationary dynamics? fully observed? known action space? What breaks if one fails?

## 4. Data

Answer the seven questions from the [datasets guide](../datasets/README.md#2-seven-questions-to-ask-of-any-world-model-dataset):

| # | Question | Answer for this design |
|---|---|---|
| 1 | Observations | |
| 2 | Actions | |
| 3 | Task signal | |
| 4 | Collector | |
| 5 | Episodes | |
| 6 | Stochasticity | |
| 7 | Scale and access | |

- **Splits:** train / validation / test, and one **off-distribution test** that the training data doesn't cover (new speed, new map, new policy, longer horizon).
- **Toy version:** the smallest dataset that still has the property the design is about, runnable on a CPU.

## 5. Hypotheses and predictions

- **H1:** <the design improves X over baseline Y on data Z>, measured by <metric>, by roughly <effect size>.
- **H2:** <the mechanism from section 1 is the reason>: removing component C should remove most of the gain.
- **What would prove it wrong:** <the result that would make you abandon or rethink the design>.

## 6. Experiments

- **Baselines, simplest first:** "repeat the last observation", a linear / Kalman / AR model, an MLP or ensemble, then the strongest published method you can run. A new design has to beat the simple baselines before it's compared against the complex ones.
- **Ablations:** remove or replace each component, one at a time.
- **Metrics** (from the [metrics table](../datasets/README.md#metrics)): at least one fidelity metric, one distribution or uncertainty metric, and one downstream metric (does it help an agent decide?).
- **Seeds and intervals:** how many seeds, and how you'll report uncertainty.
- **Compute budget:** what runs on a CPU, what needs a GPU, and how long each takes.

## 7. Results

| Method | Metric 1 | Metric 2 | Downstream | Notes |
|---|---|---|---|---|
| Baseline: repeat last | | | | |
| Baseline: | | | | |
| This design | | | | |
| − component C | | | | |

## 8. What I learned and what's next

- Which hypotheses held, which failed, and why?
- What surprised you?
- Next step: scale up, change one assumption, or abandon (and what you'd keep from it).
