from collections import defaultdict
import numpy as np
from time import perf_counter

from .oracles import BarrierL1Oracle, L1RegOracle
from .counted_oracle import CountedBarrierOracle


def _append_trace(
    history,
    oracle,
    x,
    start_time,
    fw_gap=None,
    store_x_dim=2,
    F_val=None,
    zero_frac=None,
    oracle_calls=None,
):
    if history is None:
        return
    history["time"].append(perf_counter() - start_time)
    if F_val is not None:
        history["F"].append(float(F_val))
    elif hasattr(oracle, "func"):
        history["F"].append(float(oracle.func(x)))
    if fw_gap is not None:
        history["fw_gap"].append(float(fw_gap))
    if zero_frac is not None:
        history["zero_frac"].append(float(zero_frac))
    if oracle_calls is not None:
        history["oracle_calls"].append(int(oracle_calls))
    if x.size <= store_x_dim:
        history["x"].append(np.copy(x))


def _zero_fraction(x, thresh=1e-8):
    x = np.asarray(x, dtype=float).ravel()
    return 100.0 * float(np.mean(np.abs(x) < thresh))


def _total_oracle_calls(oracle):
    if oracle is None:
        return None
    s = 0
    for name in ("n_func", "n_grad", "n_hess", "n_prox", "n_subgrad"):
        s += int(getattr(oracle, name, 0))
    return s


def _max_gamma_l1_ball(x, d, R, gamma_hi=1e6):
    """Largest gamma >= 0 with ||x + gamma * d||_1 <= R (binary search on [0, hi])."""
    x = np.asarray(x, dtype=float)
    d = np.asarray(d, dtype=float)

    def feasible(g):
        return np.linalg.norm(x + g * d, 1) <= R + 1e-12

    if not feasible(0.0):
        return 0.0
    lo = 0.0
    hi = 1.0
    while feasible(hi) and hi < gamma_hi:
        hi *= 2.0
    if feasible(hi):
        return hi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _armijo_closed_segment(oracle, x, d, gamma_max, c1=1e-4):
    """Backtracking Armijo on segment x + gamma * d, gamma in [0, gamma_max]."""
    g = oracle.grad(x)
    f0 = float(oracle.func(x))
    slope = float(np.dot(g, d))
    gamma = min(1.0, float(gamma_max))
    if gamma <= 0.0:
        return 0.0
    if slope >= 0.0:
        return 0.0
    for _ in range(60):
        if float(oracle.func(x + gamma * d)) <= f0 + c1 * gamma * slope + 1e-16:
            return gamma
        gamma *= 0.5
        if gamma * (np.linalg.norm(d) + 1e-30) < 1e-20:
            break
    return 0.0


def subgradient_method(
    oracle,
    x_0,
    tolerance=1e-5,
    max_iter=1000,
    alpha_0=1.0,
    display=False,
    trace=False,
    zero_thresh=1e-8,
):
    """
    Subgradient method; returns best iterate in the sense of smallest F(x_k)
    (lab recommendation). Stopping: relative change of iterates (template criterion).
    """
    x = np.asarray(x_0, dtype=float).ravel().copy()
    start = perf_counter()
    history = defaultdict(list) if trace else None
    best_x = x.copy()
    best_F = float(oracle.func(x))

    _append_trace(
        history,
        oracle,
        x,
        start,
        F_val=best_F,
        zero_frac=_zero_fraction(x, zero_thresh),
        oracle_calls=_total_oracle_calls(oracle),
    )

    for k in range(max_iter):
        g = oracle.subgrad(x)
        alpha = float(alpha_0) / np.sqrt(float(k + 1))
        x_new = x - alpha * g
        F_new = float(oracle.func(x_new))
        if F_new < best_F:
            best_F = F_new
            best_x = x_new.copy()

        rel = np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x))
        if display:
            print(f"subgrad k={k} rel={rel:.3e} F_best={best_F:.6e}")

        x = x_new
        _append_trace(
            history,
            oracle,
            x,
            start,
            F_val=float(oracle.func(x)),
            zero_frac=_zero_fraction(x, zero_thresh),
            oracle_calls=_total_oracle_calls(oracle),
        )

        if rel <= tolerance:
            return best_x, "success", history

    return best_x, "iterations_exceeded", history


def proximal_gradient_method(
    oracle,
    x_0,
    L_0=1.0,
    tolerance=1e-5,
    max_iter=1000,
    trace=False,
    display=False,
    zero_thresh=1e-8,
):
    """ISTA with backtracking on L (Algorithm 1 in the lab text)."""
    x = np.asarray(x_0, dtype=float).ravel().copy()
    Lk = float(L_0)
    start = perf_counter()
    history = defaultdict(list) if trace else None

    _append_trace(
        history,
        oracle,
        x,
        start,
        F_val=float(oracle.func(x)),
        zero_frac=_zero_fraction(x, zero_thresh),
        oracle_calls=_total_oracle_calls(oracle),
    )

    for k in range(max_iter):
        g = oracle.grad(x)
        Lbar = Lk
        ls = 0
        while ls < 60:
            ls += 1
            y = oracle.prox(x - (1.0 / Lbar) * g, 1.0 / Lbar)
            rhs = float(oracle.func(x) + np.dot(g, y - x) + (Lbar / 2.0) * np.linalg.norm(y - x) ** 2)
            if float(oracle.func(y)) <= rhs + 1e-12:
                break
            Lbar *= 2.0
            if Lbar > 1e12:
                break
        x_new = y
        Lk = max(L_0 * 1e-6, Lbar / 2.0)

        rel = np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x))
        if display:
            print(f"ISTA k={k} rel={rel:.3e} F={oracle.func(x_new):.6e}")

        x = x_new
        _append_trace(
            history,
            oracle,
            x,
            start,
            F_val=float(oracle.func(x)),
            zero_frac=_zero_fraction(x, zero_thresh),
            oracle_calls=_total_oracle_calls(oracle),
        )

        if rel <= tolerance:
            return x, "success", history

    return x, "iterations_exceeded", history


def proximal_fast_gradient_method(
    oracle,
    x_0,
    L_0=1.0,
    tolerance=1e-5,
    max_iter=1000,
    trace=False,
    display=False,
    zero_thresh=1e-8,
):
    """FISTA / fast composite gradient (Algorithm 2 in the lab text)."""
    x = np.asarray(x_0, dtype=float).ravel().copy()
    v = x.copy()
    A = 0.0
    Lk = float(L_0)
    start = perf_counter()
    history = defaultdict(list) if trace else None

    _append_trace(
        history,
        oracle,
        x,
        start,
        F_val=float(oracle.func(x)),
        zero_frac=_zero_fraction(x, zero_thresh),
        oracle_calls=_total_oracle_calls(oracle),
    )

    for k in range(max_iter):
        Lbar = Lk
        ls = 0
        while ls < 60:
            ls += 1
            disc = 1.0 + 4.0 * Lbar * A
            a = (1.0 + np.sqrt(max(disc, 0.0))) / (2.0 * Lbar)
            A_new = A + a
            y = (A * x + a * v) / max(A_new, 1e-30)
            g = oracle.grad(y)
            x_new = oracle.prox(y - (1.0 / Lbar) * g, 1.0 / Lbar)
            rhs = float(oracle.func(y) + np.dot(g, x_new - y) + (Lbar / 2.0) * np.linalg.norm(x_new - y) ** 2)
            if float(oracle.func(x_new)) <= rhs + 1e-12:
                break
            Lbar *= 2.0
            if Lbar > 1e12:
                break

        v = v + (A_new / max(a, 1e-30)) * (x_new - y)
        A = A_new
        Lk = max(L_0 * 1e-6, Lbar / 2.0)

        rel = np.linalg.norm(x_new - x) / max(1.0, np.linalg.norm(x))
        if display:
            print(f"FISTA k={k} rel={rel:.3e} F={oracle.func(x_new):.6e}")

        x = x_new
        _append_trace(
            history,
            oracle,
            x,
            start,
            F_val=float(oracle.func(x)),
            zero_frac=_zero_fraction(x, zero_thresh),
            oracle_calls=_total_oracle_calls(oracle),
        )

        if rel <= tolerance:
            return x, "success", history

    return x, "iterations_exceeded", history


def frank_wolfe_method(
    oracle,
    x_0,
    R,
    tolerance=1e-5,
    max_iter=1000,
    step_size_strategy="standard",
    trace=False,
    display=False,
    lambda_l1=None,
    zero_thresh=1e-8,
):
    """
    Frank–Wolfe on min f(x) s.t. ||x||_1 <= R (smooth f only in `oracle`).

    If `lambda_l1` is not None, history['F'] stores f(x)+lambda_l1*||x||_1 (composite),
    while stopping uses the Frank–Wolfe gap for f only.
    """
    x = np.asarray(x_0, dtype=float).ravel().copy()
    R = float(R)
    start = perf_counter()
    history = defaultdict(list) if trace else None
    l1_dummy = L1RegOracle(1.0)

    def F_full(xx):
        if lambda_l1 is None:
            return float(oracle.func(xx))
        return float(oracle.func(xx) + float(lambda_l1) * np.linalg.norm(xx, 1))

    s = l1_dummy.lmo(oracle.grad(x), R)
    gap = float(np.dot(oracle.grad(x), x - s))
    _append_trace(
        history,
        oracle,
        x,
        start,
        fw_gap=gap,
        F_val=F_full(x),
        zero_frac=_zero_fraction(x, zero_thresh),
        oracle_calls=_total_oracle_calls(oracle),
    )

    if gap <= tolerance:
        return x, "success", history

    for t in range(1, max_iter + 1):
        g = oracle.grad(x)
        s = l1_dummy.lmo(g, R)
        gap = float(np.dot(g, x - s))
        if gap <= tolerance:
            _append_trace(
                history,
                oracle,
                x,
                start,
                fw_gap=gap,
                F_val=F_full(x),
                zero_frac=_zero_fraction(x, zero_thresh),
                oracle_calls=_total_oracle_calls(oracle),
            )
            return x, "success", history

        if display:
            print(f"FW iter={t} f={oracle.func(x):.6e} gap={gap:.3e}")

        if step_size_strategy == "standard":
            gamma_w = (t - 1) / (t + 1)
            x = gamma_w * x + (1.0 - gamma_w) * s
        elif step_size_strategy == "armijo":
            d = s - x
            gmax = min(1.0, _max_gamma_l1_ball(x, d, R, gamma_hi=1.0))
            alpha = _armijo_closed_segment(oracle, x, d, gmax)
            x = x + alpha * d
        else:
            raise ValueError("Unknown step_size_strategy {!r}".format(step_size_strategy))

        _append_trace(
            history,
            oracle,
            x,
            start,
            fw_gap=gap,
            F_val=F_full(x),
            zero_frac=_zero_fraction(x, zero_thresh),
            oracle_calls=_total_oracle_calls(oracle),
        )

    return x, "iterations_exceeded", history


def away_step_frank_wolfe_method(
    oracle,
    x_0,
    R,
    tolerance=1e-5,
    max_iter=1000,
    step_size_strategy="armijo",
    trace=False,
    display=False,
    vertex_tol=1e-12,
    lambda_l1=None,
    zero_thresh=1e-8,
):
    x = np.asarray(x_0, dtype=float).ravel().copy()
    R = float(R)
    start = perf_counter()
    history = defaultdict(list) if trace else None
    l1_dummy = L1RegOracle(1.0)

    def F_full(xx):
        if lambda_l1 is None:
            return float(oracle.func(xx))
        return float(oracle.func(xx) + float(lambda_l1) * np.linalg.norm(xx, 1))

    def gap_fw(xx):
        gg = oracle.grad(xx)
        ss = l1_dummy.lmo(gg, R)
        return float(np.dot(gg, xx - ss)), gg, ss

    gap, _, _ = gap_fw(x)
    _append_trace(
        history,
        oracle,
        x,
        start,
        fw_gap=gap,
        F_val=F_full(x),
        zero_frac=_zero_fraction(x, zero_thresh),
        oracle_calls=_total_oracle_calls(oracle),
    )

    if gap <= tolerance:
        return x, "success", history

    for t in range(1, max_iter + 1):
        g = oracle.grad(x)
        s = l1_dummy.lmo(g, R)
        v = l1_dummy.amo(g, x, R, tol=vertex_tol)
        gap = float(np.dot(g, x - s))
        if gap <= tolerance:
            _append_trace(
                history,
                oracle,
                x,
                start,
                fw_gap=gap,
                F_val=F_full(x),
                zero_frac=_zero_fraction(x, zero_thresh),
                oracle_calls=_total_oracle_calls(oracle),
            )
            return x, "success", history

        d_fw = s - x
        d_aw = x - v
        if np.linalg.norm(d_aw, 1) <= vertex_tol:
            use_away = False
        else:
            gap_fw_dir = float(np.dot(g, x - s))
            gap_away_dir = float(np.dot(g, v - x))
            use_away = gap_fw_dir < gap_away_dir

        if use_away:
            d = d_aw
            gmax = _max_gamma_l1_ball(x, d, R)
        else:
            d = d_fw
            gmax = min(1.0, _max_gamma_l1_ball(x, d, R))

        if step_size_strategy == "standard":
            if not use_away:
                gamma_w = (t - 1) / (t + 1)
                x = gamma_w * x + (1.0 - gamma_w) * s
            else:
                gamma_a = min(gmax, 2.0 / (t + 2))
                x = x + gamma_a * d
        elif step_size_strategy == "armijo":
            alpha = _armijo_closed_segment(oracle, x, d, gmax)
            x = x + alpha * d
        else:
            raise ValueError("Unknown step_size_strategy {!r}".format(step_size_strategy))

        if display:
            mode = "away" if use_away else "fw"
            print(f"AFW iter={t} ({mode}) f={oracle.func(x):.6e} gap={gap:.3e}")

        gap, _, _ = gap_fw(x)
        _append_trace(
            history,
            oracle,
            x,
            start,
            fw_gap=gap,
            F_val=F_full(x),
            zero_frac=_zero_fraction(x, zero_thresh),
            oracle_calls=_total_oracle_calls(oracle),
        )

    return x, "iterations_exceeded", history


def _is_feasible_barrier(x, u):
    x = np.asarray(x, dtype=float).ravel()
    u = np.asarray(u, dtype=float).ravel()
    return bool(np.all(u > np.abs(x) + 1e-10))


def _newton_backtracking(barrier_oracle, z, d, g, eta=1e-4):
    """Decrease barrier objective along Newton direction with feasibility."""
    f0 = float(barrier_oracle.func(z))
    slope = float(np.dot(g, d))
    alpha = 1.0
    for _ in range(40):
        z_new = z + alpha * d
        x, u = barrier_oracle._split(z_new)
        if not _is_feasible_barrier(x, u):
            alpha *= 0.5
            continue
        f1 = float(barrier_oracle.func(z_new))
        if f1 <= f0 + eta * alpha * slope:
            return z_new, alpha
        alpha *= 0.5
    return z, 0.0


def barrier_method(
    smooth_oracle,
    x_0,
    u_0,
    lambda_reg,
    t_0=1.0,
    mu=10.0,
    tolerance_inner=1e-6,
    tolerance_outer=1e-5,
    max_iter=100,
    max_inner_iter=100,
    trace=False,
    display=False,
    newton_reg=1e-8,
    zero_thresh=1e-8,
):
    """
    Outer logarithmic barrier / interior-point scheme on the epigraph L1 model.
    Inner: inexact Newton on F_t with backtracking.

    History entries use composite F on x only: f(x) + lambda_reg * ||x||_1.
    """
    x = np.asarray(x_0, dtype=float).ravel().copy()
    u = np.asarray(u_0, dtype=float).ravel().copy()
    if x.shape != u.shape:
        raise ValueError("x_0 and u_0 must have the same shape.")
    if not _is_feasible_barrier(x, u):
        raise ValueError("Initial (x_0, u_0) must satisfy u_i > |x_i| strictly.")

    n = x.size
    t = float(t_0)
    lam = float(lambda_reg)
    start = perf_counter()
    history = defaultdict(list) if trace else None

    def composite_F(xx):
        return float(smooth_oracle.func(xx) + lam * np.linalg.norm(xx, 1))

    inner_total = 0
    aug_call_cum = 0

    def push_outer(z_vec, inner_iters, aug_calls_outer):
        xx, _uu = BarrierL1Oracle(smooth_oracle, lam, t)._split(z_vec)
        _append_trace(
            history,
            smooth_oracle,
            xx,
            start,
            F_val=composite_F(xx),
            zero_frac=_zero_fraction(xx, zero_thresh),
            oracle_calls=_total_oracle_calls(smooth_oracle),
        )
        if trace:
            history["inner_iters"].append(int(inner_iters))
            history["t_param"].append(float(t))
            history["inner_cumsum"].append(int(inner_total))
            history["aug_oracle_calls"].append(int(aug_calls_outer))

    z = np.concatenate([x, u])

    for k in range(max_iter):
        barrier_base = BarrierL1Oracle(smooth_oracle, lam, t)
        barrier = CountedBarrierOracle(barrier_base)
        inner_count = 0
        for _it in range(max_inner_iter):
            g = barrier.grad(z)
            if float(np.linalg.norm(g)) <= tolerance_inner * max(1.0, np.sqrt(2 * n)):
                break
            H = barrier.hess(z)
            H = H + newton_reg * np.eye(H.shape[0])
            try:
                d = np.linalg.solve(H, -g)
            except np.linalg.LinAlgError:
                d, *_ = np.linalg.lstsq(H, -g, rcond=None)
            z_new, alpha = _newton_backtracking(barrier, z, d, g)
            if alpha == 0.0:
                break
            z = z_new
            inner_count += 1
            inner_total += 1

        aug_call_cum += barrier.total_calls()
        x, u = barrier._split(z)
        push_outer(z, inner_count, aug_call_cum)

        if 2.0 * n / t <= tolerance_outer:
            return x, "success", history

        t *= float(mu)
        if display:
            print(f"barrier outer k={k} t={t:.3e} F={composite_F(x):.6e} inner={inner_count}")

    return x, "iterations_exceeded", history
