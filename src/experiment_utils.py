"""Helpers for lab experiments: synthetic data, λ tuning, R from dual heuristics."""

from typing import Optional

import numpy as np

from .optimization import proximal_fast_gradient_method


def synthetic_regression(m: int, n: int, rng: np.random.Generator, sparsity: float = 0.6):
    """y = A x_true + noise with approximately `sparsity` fraction of zeros in x_true."""
    A = rng.standard_normal((m, n)) / np.sqrt(max(n, 1))
    k = int(round((1.0 - sparsity) * n))
    k = max(1, min(n - 1, k))
    x_true = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    x_true[idx] = rng.standard_normal(k)
    y = A @ x_true + 0.12 * rng.standard_normal(m)
    return A, y, x_true


def synthetic_classification(m: int, n: int, rng: np.random.Generator, sparsity: float = 0.6):
    """Binary labels ±1, sparse separating direction."""
    A = rng.standard_normal((m, n)) / np.sqrt(max(n, 1))
    k = int(round((1.0 - sparsity) * n))
    k = max(1, min(n - 1, k))
    w = np.zeros(n)
    idx = rng.choice(n, size=k, replace=False)
    w[idx] = rng.standard_normal(k)
    if np.linalg.norm(w) < 1e-8:
        w[0] = 1.0
    w = 1.5 * w / np.linalg.norm(w)
    margin = A @ w
    y = np.where(margin + 0.15 * rng.standard_normal(m) >= 0.0, 1.0, -1.0)
    return A, y, w


def tune_lambda_for_zero_fraction(
    build_composite,
    n: int,
    x0: np.ndarray,
    target_low: float = 0.5,
    target_high: float = 0.8,
    max_iter_fista: int = 4000,
    zero_tol: float = 1e-12,
    rng: Optional[np.random.Generator] = None,
):
    """
    Pick λ so that a long FISTA run yields fraction of exact zeros in [target_low, target_high].
    Uses exponential search + bisection on λ (larger λ -> more zeros).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    def frac_zeros(lam):
        oracle = build_composite(float(lam))
        x, _, _ = proximal_fast_gradient_method(
            oracle,
            x0,
            L_0=1.0,
            tolerance=1e-9,
            max_iter=max_iter_fista,
            trace=False,
        )
        return float(np.mean(np.abs(x) <= zero_tol)), x

    lam = 0.05
    for _ in range(25):
        fz, _ = frac_zeros(lam)
        if fz < target_low:
            lam *= 1.35
        else:
            break
    if fz > target_high:
        lam_hi, lam_lo = lam, lam * 0.1
    else:
        lam_lo, lam_hi = lam * 0.1, lam

    for _ in range(22):
        lam_mid = np.sqrt(max(lam_lo * lam_hi, 1e-12))
        fz, x_mid = frac_zeros(lam_mid)
        if target_low <= fz <= target_high:
            return float(lam_mid), x_mid
        if fz < target_low:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid

    lam_final = float(np.sqrt(max(lam_lo * lam_hi, 1e-12)))
    _, x_best = frac_zeros(lam_final)
    return lam_final, x_best


def radius_from_reference(x_ref: np.ndarray, scale: float = 1.02) -> float:
    """Synchronize R with λ-solution: slightly inflate ||x*||_1 for constrained FW."""
    r = float(np.linalg.norm(np.asarray(x_ref, dtype=float).ravel(), 1))
    return max(r * scale, 1e-6)
