"""
Lab 0.1 -- Learning is estimation
=================================
Part A: LS channel estimation three ways (closed form / gradient descent / LMS=SGD)
        + what step size and pilot conditioning do to gradient descent
        + MAP (= ridge = LMMSE here) vs MLE when pilots are scarce
Part B: The "blur" problem -- why MSE-trained world models predict futures that never happen

numpy only. Run from the repo root:  python3 labs/phase0/lab01_learning_is_estimation.py
"""
import numpy as np

rng = np.random.default_rng(0)


def crandn(*shape):
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def qpsk(*shape):
    return (rng.choice([-1, 1], shape) + 1j * rng.choice([-1, 1], shape)) / np.sqrt(2)


def nmse_db(h_hat, h):
    return 10 * np.log10(np.sum(np.abs(h_hat - h) ** 2) / np.sum(np.abs(h) ** 2))


# ---------------------------------------------------------------------------
# Part A.1 -- the same estimate, three ways
#   model:  y = X h + n,   n ~ CN(0, s2 I)
#   loss:   L(h) = (1/N) ||y - X h||^2           (= Gaussian NLL up to constants)
#   grad:   dL/dh* = -(1/N) X^H (y - X h)       (Wirtinger gradient)
# ---------------------------------------------------------------------------
print("=" * 72)
print("PART A.1  LS closed form vs gradient descent vs LMS (= SGD, batch size 1)")
print("=" * 72)
L, N, snr_db = 4, 64, 10
s2 = 10 ** (-snr_db / 10)
h = crandn(L) / np.sqrt(L)
X = qpsk(N, L)
y = X @ h + np.sqrt(s2) * crandn(N)

h_ls = np.linalg.solve(X.conj().T @ X, X.conj().T @ y)
print(f"closed-form LS       NMSE vs true h: {nmse_db(h_ls, h):6.2f} dB")

# full-batch gradient descent
R = X.conj().T @ X / N
lam_max = np.linalg.eigvalsh(R).max()
mu = 1.0 / lam_max
h_gd = np.zeros(L, complex)
for it in range(200):
    grad = -(X.conj().T @ (y - X @ h_gd)) / N
    h_gd = h_gd - mu * grad
print(f"gradient descent     NMSE vs true h: {nmse_db(h_gd, h):6.2f} dB   "
      f"|h_gd - h_ls| = {np.linalg.norm(h_gd - h_ls):.1e}")

# LMS: one pilot at a time, several passes ("epochs") over the data
h_lms = np.zeros(L, complex)
mu_lms = 0.02
for epoch in range(50):
    for n in rng.permutation(N):
        e = y[n] - X[n] @ h_lms
        h_lms = h_lms + mu_lms * X[n].conj() * e
print(f"LMS / SGD            NMSE vs true h: {nmse_db(h_lms, h):6.2f} dB   "
      f"|h_lms - h_ls| = {np.linalg.norm(h_lms - h_ls):.1e}  (noisy: never exactly LS)")

# ---------------------------------------------------------------------------
# Part A.2 -- step size and conditioning
#   GD on a quadratic is stable iff mu < 2 / lambda_max(R)
#   convergence speed ~ condition number kappa = lambda_max / lambda_min
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PART A.2  Step size & pilot conditioning (same physics as LMS step-size rules)")
print("=" * 72)


def gd_iters_to_converge(X, y, mu, tol=1e-6, max_it=100000):
    N = X.shape[0]
    target = np.linalg.solve(X.conj().T @ X, X.conj().T @ y)
    hh = np.zeros(X.shape[1], complex)
    for it in range(1, max_it + 1):
        hh = hh + mu * (X.conj().T @ (y - X @ hh)) / N
        err = np.linalg.norm(hh - target) / np.linalg.norm(target)
        if not np.isfinite(err) or err > 1e6:
            return "diverged"
        if err < tol:
            return it
    return f">{max_it}"


for factor in [0.5, 1.0, 1.9, 2.1]:
    print(f"  well-conditioned pilots, mu = {factor:.1f}/lambda_max -> "
          f"iterations: {gd_iters_to_converge(X, y, factor / lam_max)}")

# badly designed pilots: strongly correlated (low-pass filtered) sequence
raw = qpsk(N + L)
smooth = np.convolve(raw, np.ones(6) / 6, mode="same")
X_bad = np.stack([smooth[i:i + N] for i in range(L)], axis=1)
y_bad = X_bad @ h + np.sqrt(s2) * crandn(N)
R_bad = X_bad.conj().T @ X_bad / N
ev, ev_bad = np.linalg.eigvalsh(R), np.linalg.eigvalsh(R_bad)
print(f"  condition number: good pilots {ev.max()/ev.min():6.1f}   bad pilots {ev_bad.max()/ev_bad.min():6.1f}")
print(f"  iterations at mu = 1/lambda_max: good {gd_iters_to_converge(X, y, 1/ev.max())}, "
      f"bad {gd_iters_to_converge(X_bad, y_bad, 1/ev_bad.max())}")
print("  -> neural-net training has the same disease; Adam / normalization layers are the medicine")

# ---------------------------------------------------------------------------
# Part A.3 -- MLE vs MAP when data is scarce
#   prior h ~ CN(0, (1/L) I)  ->  MAP = (X^H X + L*s2 I)^-1 X^H y   (ridge / LMMSE)
# ---------------------------------------------------------------------------
print()
print("=" * 72)
print("PART A.3  MLE (LS) vs MAP (ridge = LMMSE here), averaged over 2000 channels")
print("=" * 72)
print(f"  {'pilots N':>8} {'SNR dB':>7} {'LS NMSE':>9} {'MAP NMSE':>9}")
for N_p, snr in [(4, 0), (8, 0), (8, 10), (64, 10)]:
    s2p = 10 ** (-snr / 10)
    e_ls = e_map = p = 0.0
    for _ in range(2000):
        hh = crandn(L) / np.sqrt(L)
        Xp = qpsk(N_p, L)
        yp = Xp @ hh + np.sqrt(s2p) * crandn(N_p)
        A = Xp.conj().T @ Xp
        b = Xp.conj().T @ yp
        # lstsq, not solve: with N = L a 4x4 QPSK pilot matrix is sometimes exactly singular
        e_ls += np.sum(np.abs(np.linalg.lstsq(Xp, yp, rcond=None)[0] - hh) ** 2)
        e_map += np.sum(np.abs(np.linalg.solve(A + L * s2p * np.eye(L), b) - hh) ** 2)
        p += np.sum(np.abs(hh) ** 2)
    print(f"  {N_p:>8} {snr:>7} {10*np.log10(e_ls/p):>8.2f} {10*np.log10(e_map/p):>9.2f}")
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
print("  10 imagined futures sampled from it:", np.array2string(samples, precision=1, sign="+"))
print("  10 imagined futures from the MSE model:", np.array2string(np.full(10, pred_mse), precision=1, sign="+"))
print("  -> an MSE world model 'imagines' the car going straight through the middle: a future that never happens")
