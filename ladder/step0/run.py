"""
Step 0: given everything. EKF + MPPI on the true equations.
===========================================================
Nothing is learned yet. This step sets up the baseline every later step is
compared against, and the parts that carry forward (simulator, planner, evaluation).

Part A  Prediction: open-loop k-step error of the true model and of a wrong model
        (rod 30% too long) against the noisy simulator.
Part B  Control: swing-up with four agents, each = estimator + the same MPPI planner
          oracle      exact state, true model           (upper bound)
          diff        noisy angle, velocity by finite differences, true model
          ekf         noisy angle, EKF, true model
          ekf-wrong   noisy angle, EKF, wrong model in both filter and planner

Before running, write your predictions in ladder/step0/README.md.
Run from the repo root:  python3 -m ladder.step0.run [--episodes 5] [--plot]
"""
import argparse
import functools
import time

import numpy as np

from ladder.evaluate import evaluate_agent, print_table, rollout_error
from ladder.pendulum import EPISODE_STEPS, MEAS_STD, PROC_STD, TRUE, Params, cost, dynamics, jacobians, wrap
from ladder.planner import MPPI

WRONG = Params(l=1.3)  # the agent's belief in ekf-wrong: rod 30% too long


# ---------------------------------------------------------------------------
# Estimators: turn observations into a state estimate for the planner.
# Interface: reset(); update(y) -> x_hat; predict(u)
# ---------------------------------------------------------------------------
class Oracle:
    """The 'state' sensor already gives (theta, omega)."""
    def reset(self): pass
    def update(self, y): return np.asarray(y, dtype=float)
    def predict(self, u): pass


class FiniteDifference:
    """theta from the sensor, omega = (y_t - y_{t-1}) / dt. No model, no filtering."""
    def __init__(self, dt=TRUE.dt):
        self.dt = dt

    def reset(self):
        self.y_prev = None

    def update(self, y):
        om = 0.0 if self.y_prev is None else float(wrap(y - self.y_prev)) / self.dt
        self.y_prev = y
        return np.array([y, om])

    def predict(self, u): pass


class EKF:
    """Extended Kalman filter, angle-only measurement.

    predict:  x <- f(x, u),       P <- F P F^T + q G G^T       (q = PROC_STD^2)
    update:   nu = wrap(y - theta_hat),  S = P[0,0] + r,  K = P[:,0] / S
              x <- x + K nu,      P <- (I - K H) P (I - K H)^T + r K K^T   (Joseph form)
    NIS = nu^2 / S averages to 1 when the filter is consistent.
    """
    def __init__(self, params=TRUE, proc_std=PROC_STD, meas_std=MEAS_STD):
        self.p, self.q, self.r = params, proc_std ** 2, meas_std ** 2

    def reset(self):
        self.x = np.array([np.pi, 0.0])  # prior: hanging down, at rest
        self.P = np.diag([0.3, 0.3]) ** 2
        self.nis = []

    def update(self, y):
        nu = float(wrap(y - self.x[0]))
        S = self.P[0, 0] + self.r
        K = self.P[:, 0] / S
        IKH = np.eye(2) - np.outer(K, [1.0, 0.0])
        self.x = self.x + K * nu
        self.x[0] = wrap(self.x[0])
        self.P = IKH @ self.P @ IKH.T + self.r * np.outer(K, K)
        self.nis.append(nu * nu / S)
        return self.x.copy()

    def predict(self, u):
        F, G = jacobians(self.x, self.p)
        self.x = dynamics(self.x, u, self.p)
        self.P = F @ self.P @ F.T + self.q * np.outer(G, G)


class Agent:
    """Estimator + planner. Records its state estimates for diagnostics."""
    def __init__(self, estimator, planner):
        self.estimator, self.planner = estimator, planner

    def reset(self):
        self.estimator.reset()
        self.planner.reset()
        self.estimates = []

    def act(self, obs):
        x_hat = self.estimator.update(obs)
        self.estimates.append(x_hat)
        u = self.planner.plan(x_hat)
        self.estimator.predict(u)
        return u


def make_planner(params):
    return MPPI(functools.partial(dynamics, p=params), cost, u_max=params.u_max)


AGENTS = {
    #  name         (sensor,  estimator factory,           model params)
    "oracle":    ("state", lambda: Oracle(),               TRUE),
    "diff":      ("angle", lambda: FiniteDifference(),     TRUE),
    "ekf":       ("angle", lambda: EKF(TRUE),              TRUE),
    "ekf-wrong": ("angle", lambda: EKF(WRONG),             WRONG),
}


def omega_rmse(episode):
    """RMS error of the agent's omega estimate against the true omega."""
    est = np.array(episode["agent"].estimates)
    return float(np.sqrt(np.mean((est[:, 1] - episode["states"][:-1, 1]) ** 2)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--plot", action="store_true", help="save outputs/ladder_step0.png")
    args = ap.parse_args()

    print("=" * 72)
    print("PART A  Open-loop prediction error vs the true noisy simulator (random actions)")
    print("=" * 72)
    horizons = (1, 5, 10, 20, 40)
    errs = {name: rollout_error(functools.partial(dynamics, p=p), horizons)
            for name, p in [("true model", TRUE), ("wrong model (l=1.3)", WRONG)]}
    print_table(["model", *[f"k={k}" for k in horizons]],
                [[name, *[f"{e[k][0]:.3f}" for k in horizons]] for name, e in errs.items()])
    print("theta RMSE [rad] after k steps (dt = 0.05 s, so k=40 is 2 s)\n")

    print("=" * 72)
    print(f"PART B  Swing-up, {args.episodes} episodes x {EPISODE_STEPS * TRUE.dt:.0f} s, same seeds for every agent")
    print("=" * 72)
    results, rows = {}, []
    for name, (sensor, make_est, params) in AGENTS.items():
        t0 = time.time()
        res = evaluate_agent(lambda: Agent(make_est(), make_planner(params)),
                             n_episodes=args.episodes, sensor=sensor)
        results[name] = res
        eps = res["episodes"]
        nis = [np.mean(e["agent"].estimator.nis) for e in eps] if name.startswith("ekf") else None
        rows.append([name, f"{res['ret_mean']:8.1f} ± {res['ret_std']:5.1f}",
                     f"{100 * res['upright']:5.1f}",
                     f"{np.mean([omega_rmse(e) for e in eps]):.3f}",
                     f"{np.mean(nis):.2f}" if nis else "—",
                     f"{time.time() - t0:.1f}"])
    print_table(["agent", "return", "upright % (last 5 s)", "omega est. RMSE", "mean NIS", "time [s]"], rows)
    print("NIS = normalized innovation squared; a consistent filter averages 1.")

    if args.plot:
        plot(results)


def plot(results):
    import os

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for name, res in results.items():
        ep = res["episodes"][0]
        t = np.arange(len(ep["states"])) * TRUE.dt
        ax[0].plot(t, np.abs(wrap(ep["states"][:, 0])), label=name)
        ax[2].plot(t[:-1], ep["actions"], label=name, lw=0.8)
    for name in ("diff", "ekf"):
        ep = results[name]["episodes"][0]
        t = np.arange(len(ep["states"])) * TRUE.dt
        ax[1].plot(t[:-1], np.array(ep["agent"].estimates)[:, 1], label=f"{name} estimate", lw=0.8)
    ax[1].plot(t, results["ekf"]["episodes"][0]["states"][:, 1], "k", label="true", lw=1.2)
    ax[0].set_ylabel("|theta| [rad]  (0 = up)")
    ax[1].set_ylabel("omega [rad/s]")
    ax[2].set_ylabel("torque u [N m]")
    ax[2].set_xlabel("time [s]")
    for a in ax:
        a.legend(loc="upper right", fontsize=8)
        a.grid(alpha=0.3)
    fig.suptitle("Step 0, episode 0")
    fig.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    fig.savefig("outputs/ladder_step0.png", dpi=120)
    print("saved outputs/ladder_step0.png")


if __name__ == "__main__":
    main()
