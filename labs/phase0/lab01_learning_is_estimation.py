"""
Lab 0.1 -- Learning is estimation
=================================
Part A: Linear regression three ways (closed form / gradient descent / minibatch SGD)
        + what step size and feature scaling do to gradient descent
        + MAP (= ridge regression) vs MLE when data is scarce: polynomial overfitting
Part B: The "blur" problem -- why MSE-trained world models predict futures that never happen

numpy only. Run from the repo root:  python3 labs/phase0/lab01_learning_is_estimation.py
"""
import numpy as np

rng = np.random.default_rng(0)


def rmse(pred, y):
    return np.sqrt(np.mean((pred - y) ** 2))


def rel_dist(w, w_ref):
    return np.linalg.norm(w - w_ref) / np.linalg.norm(w_ref)


# ---------------------------------------------------------------------------
# A toy house-price dataset (prices in thousands of euros)
#   features: area (m^2), number of rooms, age (years)
#   model:    price = w0 + w1*area + w2*rooms + w3*age + noise,  noise ~ N(0, 25^2)
# ---------------------------------------------------------------------------
def make_houses(n):
    area = rng.uniform(40, 250, n)
    rooms = np.clip(np.round(area / 35 + rng.normal(0, 0.8, n)), 1, 8)
    age = rng.uniform(0, 80, n)
    price = 50 + 3.0 * area + 8.0 * rooms - 0.8 * age + 25 * rng.standard_normal(n)
    return np.stack([area, rooms, age], axis=1), price


F_train, y = make_houses(500)
F_test, y_test = make_houses(2000)
mean, std = F_train.mean(0), F_train.std(0)


def design(F, standardize):
    Z = (F - mean) / std if standardize else F
    return np.hstack([np.ones((len(F), 1)), Z])        # first column = intercept


X, X_test = design(F_train, True), design(F_test, True)
N = len(y)

# ---------------------------------------------------------------------------
# Part A.1 -- the same estimate, three ways
#   loss:   L(w) = (1/2N) ||y - X w||^2        (= Gaussian NLL up to constants)
#   grad:   dL/dw = -(1/N) X^T (y - X w)
# ---------------------------------------------------------------------------
print("=" * 72)
print("PART A.1  Linear regression: closed form vs gradient descent vs minibatch SGD")
print("=" * 72)
w_ls = np.linalg.solve(X.T @ X, X.T @ y)                # normal equations


def report(name, w):
    gap = "" if w is w_ls else f"   distance to closed form {rel_dist(w, w_ls):.0e}"
    print(f"  {name:<34} test RMSE {rmse(X_test @ w, y_test):6.2f} k€{gap}")


report("closed form (normal equations)", w_ls)

R = X.T @ X / N                                         # Hessian of L
lam_max = np.linalg.eigvalsh(R).max()
w_gd = np.zeros(X.shape[1])
for it in range(500):
    w_gd = w_gd + (1.0 / lam_max) * X.T @ (y - X @ w_gd) / N
report("gradient descent, 500 steps", w_gd)


def sgd(batch, lr0, steps=10_000, decay=False):
    """Minibatch SGD; with decay, the learning rate shrinks as lr0 / (1 + k/2000)."""
    w = np.zeros(X.shape[1])
    k = 0
    while k < steps:
        order = rng.permutation(N)                      # one epoch = one pass in random order
        for start in range(0, N, batch):
            if k == steps:
                break
            b = order[start:start + batch]
            lr = lr0 / (1 + k / 2000) if decay else lr0
            w = w + lr * X[b].T @ (y[b] - X[b] @ w) / len(b)
            k += 1
    return w


for batch, decay, label in [(1, False, "batch  1, constant lr"),
                            (32, False, "batch 32, constant lr"),
                            (1, True, "batch  1, decaying lr")]:
    report(f"SGD, {label}", sgd(batch, 0.05, decay=decay))
print("  (SGD: 10,000 steps each; distance = |w - w_ls| / |w_ls|)")
print("  -> same answer; SGD keeps jittering around it, less with bigger batches or a decaying learning rate")
print(f"  learned weights (standardized features): intercept {w_ls[0]:.1f}, area {w_ls[1]:.1f}, "
      f"rooms {w_ls[2]:.1f}, age {w_ls[3]:.1f}")

# ---------------------------------------------------------------------------
# Part A.2 -- step size and feature scaling
#   GD on a quadratic is stable iff lr < 2 / lambda_max(R)
#   convergence speed ~ condition number kappa = lambda_max / lambda_min
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PART A.2  Step size & feature scaling")
print("=" * 72)


def gd_iters_to_converge(X, y, lr, tol=1e-6, max_it=200_000):
    N = X.shape[0]
    target = np.linalg.solve(X.T @ X, X.T @ y)
    w = np.zeros(X.shape[1])
    for it in range(1, max_it + 1):
        w = w + lr * X.T @ (y - X @ w) / N
        err = np.linalg.norm(w - target) / np.linalg.norm(target)
        if not np.isfinite(err) or err > 1e6:
            return "diverges"
        if err < tol:
            return it
    return f">{max_it:,}"


for factor in [0.5, 1.0, 1.9, 2.1]:
    print(f"  standardized features, lr = {factor:.1f}/lambda_max -> "
          f"iterations: {gd_iters_to_converge(X, y, factor / lam_max)}")

X_raw = design(F_train, False)
ev = np.linalg.eigvalsh(R)
ev_raw = np.linalg.eigvalsh(X_raw.T @ X_raw / N)
print(f"  condition number: standardized {ev.max() / ev.min():9.1f}   raw (m^2, rooms, years) "
      f"{ev_raw.max() / ev_raw.min():9.1f}")
print(f"  iterations at lr = 1/lambda_max: standardized {gd_iters_to_converge(X, y, 1 / ev.max())}, "
      f"raw {gd_iters_to_converge(X_raw, y, 1 / ev_raw.max())}")
print("  -> always standardize inputs; in deep nets, normalization layers and Adam treat the same problem")

# ---------------------------------------------------------------------------
# Part A.3 -- MLE vs MAP when data is scarce: polynomial curve fitting
#   y = sin(2 pi x) + noise,  noise ~ N(0, 0.3^2),  degree-9 polynomial features
#   prior w ~ N(0, tau^2 I)  ->  MAP = (Phi^T Phi + (s^2/tau^2) I)^-1 Phi^T y   (ridge)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PART A.3  MLE (least squares) vs MAP (ridge): degree-9 polynomial, 500 random datasets")
print("=" * 72)
DEG, NOISE, TAU = 9, 0.3, 10.0
lam = NOISE ** 2 / TAU ** 2


def poly(x):
    return np.vander(x, DEG + 1, increasing=True)      # columns 1, x, x^2, ..., x^9


x_grid = np.linspace(0, 1, 500)
f_grid = np.sin(2 * np.pi * x_grid)
print(f"  prior: each weight ~ N(0, {TAU:.0f}^2)  ->  ridge lambda = {lam:.1e}")
print(f"  {'points N':>8} {'MLE RMSE':>9} {'MAP RMSE':>9} {'MLE max|w|':>11} {'MAP max|w|':>11}")
for n in [10, 15, 30, 100]:
    e_ml, e_map, w_ml_max, w_map_max = [], [], [], []
    for _ in range(500):
        x = rng.uniform(0, 1, n)
        t = np.sin(2 * np.pi * x) + NOISE * rng.standard_normal(n)
        P = poly(x)
        w_ml = np.linalg.lstsq(P, t, rcond=None)[0]
        w_map = np.linalg.solve(P.T @ P + lam * np.eye(DEG + 1), P.T @ t)
        e_ml.append(rmse(poly(x_grid) @ w_ml, f_grid))
        e_map.append(rmse(poly(x_grid) @ w_map, f_grid))
        w_ml_max.append(np.abs(w_ml).max())
        w_map_max.append(np.abs(w_map).max())
    print(f"  {n:>8} {np.median(e_ml):>9.2f} {np.median(e_map):>9.2f} "
          f"{np.median(w_ml_max):>11.1e} {np.median(w_map_max):>11.1f}")
print("  (medians over datasets; RMSE measured against the true curve sin(2 pi x) on [0, 1])")
print("  -> the prior (= regularizer, 'weight decay') matters most when data is scarce")

# ---------------------------------------------------------------------------
# Part B -- the blur problem
#   A car reaches an intersection. Next lateral position: -1 (left) or +1 (right), 50/50.
#   Fit three "world models" by maximum likelihood with different noise assumptions.
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PART B  The blur problem: what does each loss predict for a two-way future?")
print("=" * 72)
M = 5000
y_future = rng.choice([-1.0, 1.0], M) + 0.1 * rng.standard_normal(M)

pred_mse = y_future.mean()          # MLE under Gaussian noise -> mean
pred_l1 = np.median(y_future)       # MLE under Laplace noise  -> median


def frac_realistic(pred):
    return np.mean(np.abs(y_future - pred) < 0.3)


print(f"  Gaussian NLL (MSE) predicts {pred_mse:+.3f}  -> {100*frac_realistic(pred_mse):4.1f}% of real futures look like this")
print(f"  Laplace  NLL (L1)  predicts {pred_l1:+.3f}  -> {100*frac_realistic(pred_l1):4.1f}% of real futures look like this")

# categorical over bins: MLE is just the normalized histogram
edges = np.linspace(-1.6, 1.6, 17)
counts, _ = np.histogram(y_future, edges)
probs = counts / counts.sum()
print("  Categorical NLL (cross-entropy over 16 bins) learns the whole distribution:")
centers = (edges[:-1] + edges[1:]) / 2
for c, pr in zip(centers, probs):
    print(f"    {c:+.1f} | {'#' * int(round(pr * 120))}")
samples = rng.choice(centers, 10, p=probs)
print("  10 imagined futures sampled from it:   ", np.array2string(samples, precision=1, sign="+", floatmode="fixed"))
print("  10 imagined futures from the MSE model:", np.array2string(np.full(10, pred_mse), precision=1, sign="+", floatmode="fixed"))
print("  -> an MSE world model 'imagines' the car going straight through the middle: a future that never happens")
