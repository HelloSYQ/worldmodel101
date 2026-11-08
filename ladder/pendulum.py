"""
The world for the whole ladder: a torque-limited pendulum that you own.

Because you wrote the simulator, you always know the true dynamics. Every step
can therefore check its model against ground truth, not only by task return.

State    x = (theta, omega).  theta = 0 is upright, theta = pi hangs down.
Action   u = torque, clipped to [-u_max, u_max].  u_max < m g l, so the motor
         cannot lift the pendulum directly: it has to swing it up.
Dynamics semi-implicit Euler with step dt:
           omega' = omega + dt * ( (g/l) sin(theta) - b/(m l^2) omega + (u + w)/(m l^2) )
           theta' = theta + dt * omega'
         w ~ N(0, PROC_STD^2) is a random torque disturbance (process noise).
Sensors  'state'  -> (theta, omega), exact           (oracle, for baselines)
         'angle'  -> wrap(theta + v), v ~ N(0, MEAS_STD^2)
         'pixels' -> added in step 3
Cost     c(x, u) = wrap(theta)^2 + 0.1 omega^2 + 0.001 u^2      (reward = -cost)

Every function is batched over leading axes: x has shape (..., 2), u has shape (...).
"""
from dataclasses import dataclass

import numpy as np

PROC_STD = 0.1    # torque disturbance std [N m]
MEAS_STD = 0.05   # angle sensor noise std [rad] (about 3 degrees)
EPISODE_STEPS = 300  # 15 s at dt = 0.05: about 7 s to swing up, then balance


@dataclass(frozen=True)
class Params:
    g: float = 9.81     # gravity [m/s^2]
    m: float = 1.0      # point mass at the tip [kg]
    l: float = 1.0      # rod length [m]
    b: float = 0.1      # viscous damping [N m s]
    dt: float = 0.05    # time step [s]
    u_max: float = 2.0  # torque limit [N m]; m g l = 9.81, so it must swing


TRUE = Params()


def wrap(a):
    """Map angles to [-pi, pi)."""
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def dynamics(x, u, p=TRUE, w=0.0):
    """One step of the true equations. w is the torque disturbance (0 = mean dynamics)."""
    x = np.asarray(x, dtype=float)
    u = np.clip(u, -p.u_max, p.u_max)
    th, om = x[..., 0], x[..., 1]
    ml2 = p.m * p.l ** 2
    om_next = om + p.dt * (p.g / p.l * np.sin(th) - p.b / ml2 * om + (u + w) / ml2)
    th_next = th + p.dt * om_next
    return np.stack([th_next, om_next], axis=-1)


def jacobians(x, p=TRUE):
    """F = dx'/dx and G = dx'/dw at a single state x (for the EKF).

    The dynamics are linear in u, so F does not depend on the action.
    """
    dt, ml2 = p.dt, p.m * p.l ** 2
    a = dt * p.g / p.l * np.cos(x[0])  # d omega' / d theta
    c = 1.0 - dt * p.b / ml2           # d omega' / d omega
    F = np.array([[1.0 + dt * a, dt * c],
                  [a, c]])
    G = np.array([dt * dt / ml2, dt / ml2])
    return F, G


def cost(x, u):
    x = np.asarray(x)
    return wrap(x[..., 0]) ** 2 + 0.1 * x[..., 1] ** 2 + 0.001 * np.asarray(u) ** 2


class Pendulum:
    """The environment. `state` is ground truth: agents must not read it, evaluation may."""

    def __init__(self, sensor="angle", proc_std=PROC_STD, meas_std=MEAS_STD, params=TRUE, seed=0):
        if sensor not in ("state", "angle"):
            raise ValueError(f"unknown sensor {sensor!r}")
        self.sensor, self.proc_std, self.meas_std, self.params = sensor, proc_std, meas_std, params
        self.rng = np.random.default_rng(seed)
        self.state = None

    def reset(self):
        """Start hanging down, with a small random offset."""
        self.state = np.array([np.pi, 0.0]) + 0.1 * self.rng.standard_normal(2)
        return self._observe()

    def step(self, u):
        u = float(np.clip(u, -self.params.u_max, self.params.u_max))
        reward = -float(cost(self.state, u))
        w = self.proc_std * self.rng.standard_normal()
        self.state = dynamics(self.state, u, self.params, w)
        return self._observe(), reward

    def _observe(self):
        # Always draw the sensor noise, so the same seed gives every agent the
        # same disturbance and noise sequence, whatever the sensor (common random numbers).
        v = self.meas_std * self.rng.standard_normal()
        if self.sensor == "state":
            return self.state.copy()
        return float(wrap(self.state[0] + v))
