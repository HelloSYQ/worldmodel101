# worldmodel101

A self-paced course on **world models**, from first principles to current research.

- **Goal:** research-level understanding, ending in an original capstone project
- **Pace:** ~5 h/week, about 21–22 weeks (we'll revisit the pace as we go)
- **Assumed background:** NR / wireless (estimation, Kalman filtering, linear algebra). No prior ML/RL training experience.

## The organizing grid

A world model answers *"what happens next if I do $a$?"*, i.e. it models

```math
p_\theta(s_{t+1}\mid s_t, a_t), \qquad p_\theta(o_t\mid s_t), \qquad p_\theta(r_t\mid s_t, a_t)
```

and an agent uses it to predict, imagine, plan or simulate. Every paper in this course gets placed on two axes:

| What the model predicts | What it is used for |
|---|---|
| raw observations (pixels / video) | planning (MPC, tree search) |
| latents trained with reconstruction | training a policy in imagination |
| latents trained without reconstruction | acting as an environment / simulator |
| only task quantities (reward, value, policy) | learning representations |
| | generating data, evaluating policies |

## How each lesson runs

1. **Intuition:** why the idea exists, what problem it fixes
2. **Math:** derived, not asserted (LaTeX in `notes/`)
3. **Paper walkthrough**
4. **Lab:** small CPU-sized experiment in `labs/`
5. **Check questions:** answered in `answers/` before moving on

## Repository layout

```
notes/     one Markdown file per lesson (math in LaTeX; GitHub renders it)
labs/      runnable code, one folder per phase
papers/    three-pass reading notes, each placed on the grid above
answers/   check-question answers + feedback
```

Setup: `pip install -r requirements.txt`, then run labs from the repo root, e.g. `python3 labs/phase0/lab01_learning_is_estimation.py`.

---

## Syllabus and progress

### Phase 0: Prerequisites (weeks 1–5)

| Week | Lesson | Lab | Status |
|---|---|---|---|
| 1 | [0.1 Learning is estimation](notes/0.1-learning-is-estimation.md): MLE, KL, loss functions, MAP, gradient descent | LS / GD / LMS, conditioning, MLE vs MAP, the blur problem | 🟡 check questions pending |
| 1 | 0.2 Neural networks and PyTorch: MLP, backprop/autograd, Adam, generalization | Learn a nonlinear PA model (Rapp/Saleh) with an MLP; learned-variance head | ⬜ |
| 2 | 0.3 Latent variables: EM → VAE (ELBO, reparameterization) | VAE on MNIST | ⬜ |
| 3 | 0.4 Sequence models: RNN/GRU → attention → Transformer | Jakes-fading channel prediction: Kalman/AR vs GRU vs Transformer | ⬜ |
| 4 | 0.5 RL I: MDPs, Bellman, value iteration, Q-learning | Gridworld | ⬜ |
| 4 | 0.6 RL II: policy gradient, actor-critic, POMDPs | Actor-critic on CartPole | ⬜ |
| 5 | Review, checkpoint quiz, first full paper: Ha & Schmidhuber (2018) | — | ⬜ |

### Phase 1: What is a world model? Classical roots (1 week) ⬜
Craik's mental models, Dyna, model-based vs model-free RL; state-space models, system identification, Kalman filter + MPC; anatomy of a modern world model.
*Lab:* pendulum with Kalman + MPC, then replace the dynamics with a learned MLP and plan with CEM.

### Phase 2: Learning dynamics in state space (1 week) ⬜
PILCO, PETS (ensembles; epistemic vs aleatoric uncertainty), MBPO; compounding error, model exploitation, objective mismatch.
*Lab:* ensemble dynamics model on cartpole; error vs rollout horizon.

### Phase 3: Latent world models from pixels: the Dreamer line (2 weeks, core) ⬜
Ha & Schmidhuber (2018); PlaNet and the RSSM (sequential ELBO derived); DreamerV1 → V2 → V3; Dreamer 4 (2025).
*Lab:* mini-RSSM on a pixel environment; visualize dreams and drift.

### Phase 4: Decision-aware models (1 week) ⬜
MuZero, EfficientZero (V2); value equivalence, SPR, bisimulation; TD-MPC2.
*Lab:* tiny MuZero on a grid game or a TD-MPC-style planner.

### Phase 5: Sequence models and tokens (1 week) ⬜
IRIS, STORM, Δ-IRIS; Trajectory / Decision Transformer; DIAMOND; VQ-VAE, FSQ, video tokenizers.

### Phase 6: Generative world simulators (2 weeks, core) ⬜
Latent diffusion, DiT, flow matching; Sora's "world simulator" claim; UniSim, GameNGen, Oasis, WHAM/Muse, Matrix-Game 2.0; Genie 1 → 2 → 3; GAIA-1/2, Cosmos; drift and exposure bias (Diffusion Forcing, Self Forcing), real-time distillation (CausVid), long-horizon memory, controllability.
*Lab:* small action-conditioned diffusion world model on a toy 2D game; drift with and without diffusion-forcing noise.

### Phase 7: Predictive-representation world models: JEPA (1 week) ⬜
LeCun (2022); collapse prevention (EMA, VICReg); I-JEPA → V-JEPA → V-JEPA 2 / V-JEPA 2-AC; DINO-WM; generative vs non-generative debate.
*Lab:* latent MPC on frozen pretrained features.

### Phase 8: Embodied world models (1 week) ⬜
World models as policy (UniPi), data engine (DreamGen, Cosmos), policy evaluator; UWM, GR-series, LAPA, Navigation World Models; NeRF / 3D Gaussian splatting, World Labs' Marble.

### Phase 9: LLMs and world models (1 week) ⬜
Othello-GPT, linear probes of space/time; Vafa et al. (2024, 2025); RAP, web-agent world models, Code World Model; Richens et al., *General agents need world models* (ICML 2025).

### Phase 10: Frontier and research practice (ongoing) ⬜
Evaluation (Physics-IQ, physical-law studies, consistency, controllability, downstream usefulness); open problems (memory, causality, uncertainty, hierarchy, scaling, real-time cost, learned vs explicit physics); live literature sweep for work after mid-2026.

### Track W: World models for the radio channel (capstone) ⬜
Channel / UE mobility as environment, CSI-RS / SSB measurements as observations, beam / MCS as actions. TR 38.843 use cases (CSI prediction, BM-Case1/2, CSI compression, positioning), LCM and model monitoring as world-model drift; Rel-19 and 6G AI-native direction; ray-tracing digital twins vs learned dynamics.
*Capstone:* RSSM-style model on TR 38.901 CDL/TDL channel sequences, predicting CSI / beam quality $k$ slots ahead and used for beam selection, vs Kalman / AR baselines.

---

## Research thread (runs through every phase)

- **Weekly:** one paper read with the three-pass method ([template](papers/_template.md)), placed on the grid.
- **Each core phase:** reproduce one result at toy scale.
- **Capstone:** a small original project (default: Track W).

### Reading log

| Week | Reading | Notes |
|---|---|---|
| 1 | Keshav, *How to Read a Paper*; Murphy PML1 §4.2, §6.1–6.2 | — |
