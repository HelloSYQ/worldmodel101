"""
MPPI: Model Predictive Path Integral control (Williams et al., 2017), simplified.

The planner sees the world only through two batched functions:

    model(x, u) -> x_next      x: (K, n), u: (K,)
    cost(x, u)  -> c           shape (K,)

Each step of the ladder swaps in a different `model` (true equations, learned
MLP, ensemble, latent RSSM) and leaves this file unchanged. That's what makes
results comparable from one step to the next.

Each call to plan(x):
  1. sample K action sequences around the current plan:  U_k = mean + eps_k
  2. roll each one out through the model and add up its cost S_k
  3. reweight:  w_k ∝ exp(-(S_k - min S) / temperature)   (a soft argmin)
  4. new plan = sum_k w_k U_k;  apply its first action and shift it one step (warm start)
"""
import numpy as np


class MPPI:
    def __init__(self, model, cost, horizon=30, n_samples=500, noise_std=1.0,
                 temperature=1.0, u_max=2.0, seed=0):
        self.model, self.cost = model, cost
        self.horizon, self.n_samples = horizon, n_samples
        self.noise_std, self.temperature, self.u_max = noise_std, temperature, u_max
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.mean = np.zeros(self.horizon)

    def plan(self, x):
        K, H = self.n_samples, self.horizon
        U = self.mean + self.noise_std * self.rng.standard_normal((K, H))
        U[0] = self.mean  # always keep the current plan as a candidate
        U = np.clip(U, -self.u_max, self.u_max)

        xs = np.repeat(np.asarray(x, dtype=float)[None], K, axis=0)
        S = np.zeros(K)
        for t in range(H):
            S += self.cost(xs, U[:, t])
            xs = self.model(xs, U[:, t])

        w = np.exp(-(S - S.min()) / self.temperature)
        w /= w.sum()
        self.mean = w @ U

        u = float(self.mean[0])
        self.mean = np.append(self.mean[1:], self.mean[-1])
        return u
