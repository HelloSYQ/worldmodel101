"""
Shared evaluation. Every step reports the same two numbers, on the same seeds:

1. Control:    episode return and % of the last 5 s spent upright,
               from evaluate_agent(make_agent).
2. Prediction: open-loop k-step error of the model against the true noisy
               simulator, from rollout_error(model).

An agent has two methods: reset() and act(obs) -> u.
A model is a batched function model(x, u) -> x_next (the same one the planner uses).
From step 3 the model lives in a latent space; rollout_error will then gain a
readout from latent to state.
"""
import numpy as np

from ladder.pendulum import EPISODE_STEPS, PROC_STD, TRUE, Pendulum, dynamics, wrap

UPRIGHT_TOL = 0.3  # |theta| below this counts as upright [rad], about 17 degrees


def run_episode(agent, env, steps=EPISODE_STEPS):
    obs = env.reset()
    agent.reset()
    states, actions, rewards = [env.state.copy()], [], []
    for _ in range(steps):
        u = agent.act(obs)
        obs, r = env.step(u)
        states.append(env.state.copy())
        actions.append(u)
        rewards.append(r)
    states = np.array(states)
    upright = np.abs(wrap(states[-100:, 0])) < UPRIGHT_TOL
    return dict(ret=float(np.sum(rewards)), upright=float(upright.mean()),
                states=states, actions=np.array(actions), agent=agent)


def evaluate_agent(make_agent, n_episodes=5, **env_kwargs):
    """Run n_episodes, each with a fresh agent and env seed 0..n-1 (the same seeds for every step)."""
    episodes = [run_episode(make_agent(), Pendulum(seed=i, **env_kwargs)) for i in range(n_episodes)]
    rets = np.array([e["ret"] for e in episodes])
    return dict(ret_mean=float(rets.mean()), ret_std=float(rets.std()),
                upright=float(np.mean([e["upright"] for e in episodes])), episodes=episodes)


def rollout_error(model, horizons=(1, 5, 10, 20, 40), n_starts=500, proc_std=PROC_STD,
                  params=TRUE, seed=1234):
    """RMS open-loop prediction error after k steps, under random actions.

    Start states are uniform over theta in [-pi, pi) and omega in [-4, 4]. The model
    and the true simulator (with process noise) get the same start and action sequence.
    Returns {k: (theta_rmse [rad], omega_rmse [rad/s])}.
    """
    rng = np.random.default_rng(seed)
    H = max(horizons)
    x0 = np.column_stack([rng.uniform(-np.pi, np.pi, n_starts), rng.uniform(-4, 4, n_starts)])
    U = rng.uniform(-params.u_max, params.u_max, (n_starts, H))
    W = proc_std * rng.standard_normal((n_starts, H))

    x_true, x_pred, out = x0.copy(), x0.copy(), {}
    for t in range(H):
        x_true = dynamics(x_true, U[:, t], params, W[:, t])
        x_pred = model(x_pred, U[:, t])
        if t + 1 in horizons:
            e_th = wrap(x_pred[:, 0] - x_true[:, 0])
            e_om = x_pred[:, 1] - x_true[:, 1]
            out[t + 1] = (float(np.sqrt(np.mean(e_th ** 2))), float(np.sqrt(np.mean(e_om ** 2))))
    return out


def print_table(headers, rows):
    rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.rjust(w) if i else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    print(line(headers))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print(line(r))
