import numpy as np
import scipy
from scipy.special import expit

_LOG2 = np.log(2.0)
_EXP_CLIP = 709.0


class BaseSmoothOracle(object):
    """Base class for smooth function."""

    def func(self, x):
        raise NotImplementedError("Func is not implemented.")

    def grad(self, x):
        raise NotImplementedError("Grad is not implemented.")

    def hess(self, x):
        raise NotImplementedError("Hess is not implemented.")


class BaseProxOracle(object):
    """Base class for proximal h(x)-part in a composite function f(x) + h(x)."""

    def func(self, x):
        raise NotImplementedError("Func is not implemented.")

    def prox(self, x, alpha):
        raise NotImplementedError("Prox is not implemented.")


class BaseCompositeOracle(object):
    """phi(x) := f(x) + h(x), where f is a smooth part, h is a simple part."""

    def __init__(self, f, h):
        self._f = f
        self._h = h

    @property
    def smooth(self):
        return self._f

    @property
    def nonsmooth_simple(self):
        return self._h

    def func(self, x):
        return self._f.func(x) + self._h.func(x)

    def grad(self, x):
        return self._f.grad(x)

    def prox(self, x, alpha):
        return self._h.prox(x, alpha)


class BaseNonsmoothConvexOracle(object):
    """Base class for implementation of oracle for nonsmooth convex function."""

    def func(self, x):
        raise NotImplementedError("Func is not implemented.")

    def subgrad(self, x):
        raise NotImplementedError("Subgrad is not implemented.")


class L1RegOracle(BaseProxOracle):
    """
    Oracle for L1-regularizer h(x) = regcoef * ||x||_1.
    Also provides LMO / AMO on the L1-ball of radius R for Frank–Wolfe variants.
    """

    def __init__(self, regcoef):
        self.regcoef = float(regcoef)

    def func(self, x):
        return self.regcoef * np.linalg.norm(x, 1)

    def prox(self, x, alpha):
        """Soft-thresholding: prox_{alpha * h}(x) with h = regcoef * ||.||_1."""
        t = alpha * self.regcoef
        return np.sign(x) * np.maximum(np.abs(x) - t, 0.0)

    @staticmethod
    def lmo(grad, radius):
        """
        Linear minimization oracle on the L1-ball:
            y = argmin_{||y||_1 <= R} <grad, y>.
        Optimum is at a vertex -R * sign(grad_i) * e_i for i = argmax |grad_i|.
        """
        g = np.asarray(grad, dtype=float).ravel()
        n = g.size
        if n == 0:
            return np.array([])
        i = int(np.argmax(np.abs(g)))
        y = np.zeros(n, dtype=float)
        if g[i] == 0.0:
            y[i] = -float(radius)
        else:
            y[i] = -float(radius) * np.sign(g[i])
        return y

    @staticmethod
    def amo(grad, x, radius, tol=1e-12):
        """
        Away-step oracle: vertex v on the minimal face of ||.||_1 <= R containing x
        that maximizes <grad, v> (most "aligned" with the gradient among face vertices).

        For ||x||_1 < R the minimal face is the whole cross-polytope; v is a global
        maximizer over all vertices ±R e_i.
        For ||x||_1 ≈ R we restrict to vertices sign(x_i) R e_i on the support of x.
        """
        g = np.asarray(grad, dtype=float).ravel()
        x = np.asarray(x, dtype=float).ravel()
        n = g.size
        R = float(radius)
        nx = np.linalg.norm(x, 1)

        def best_over_vertices(idxs):
            best_val = -np.inf
            best = np.zeros(n, dtype=float)
            for j in idxs:
                for sgn in (-1.0, 1.0):
                    v = np.zeros(n, dtype=float)
                    v[j] = sgn * R
                    val = float(np.dot(g, v))
                    if val > best_val:
                        best_val = val
                        best = v
            return best

        if nx < R - 1e-9:
            idxs = np.arange(n)
            return best_over_vertices(idxs)

        supp = np.where(np.abs(x) > tol)[0]
        if supp.size == 0:
            return best_over_vertices(np.arange(n))

        best_val = -np.inf
        best_v = np.zeros(n, dtype=float)
        for j in supp:
            v = np.zeros(n, dtype=float)
            v[j] = R * np.sign(x[j])
            val = float(np.dot(g, v))
            if val > best_val:
                best_val = val
                best_v = v
        return best_v


class BarrierL1Oracle(object):
    """
    Logarithmic barrier for the epigraph reformulation of L1:
        F_t(x, u) = t * ( f(x) + lambda * sum_i u_i )
                    - sum_i log(u_i - x_i) - sum_i log(u_i + x_i),
    domain: u_i > |x_i| (strict feasibility).
    """

    def __init__(self, smooth_oracle, lambda_reg, t):
        self.smooth_oracle = smooth_oracle
        self.lambda_reg = float(lambda_reg)
        self.t = float(t)

    def _split(self, z):
        z = np.asarray(z, dtype=float).ravel()
        n = z.size // 2
        return z[:n], z[n:]

    def func(self, z):
        x, u = self._split(z)
        if np.any(u <= np.abs(x) + 1e-14):
            return np.inf
        lin = self.lambda_reg * np.sum(u)
        bar = -np.sum(np.log(u - x) + np.log(u + x))
        return self.t * (self.smooth_oracle.func(x) + lin) + bar

    def grad(self, z):
        x, u = self._split(z)
        a = u - x
        b = u + x
        gx = self.t * self.smooth_oracle.grad(x) + 1.0 / a - 1.0 / b
        gu = self.t * self.lambda_reg - 1.0 / a - 1.0 / b
        return np.concatenate([gx, gu])

    def hess(self, z):
        x, u = self._split(z)
        n = x.size
        a = u - x
        b = u + x
        inv_a2 = 1.0 / (a * a)
        inv_b2 = 1.0 / (b * b)
        d_xx = inv_a2 + inv_b2
        d_uu = inv_a2 + inv_b2
        d_xu = -inv_a2 + inv_b2
        Hf = self.smooth_oracle.hess(x)
        H = np.zeros((2 * n, 2 * n), dtype=float)
        H[:n, :n] = self.t * Hf + np.diag(d_xx)
        H[:n, n:] = np.diag(d_xu)
        H[n:, :n] = np.diag(d_xu)
        H[n:, n:] = np.diag(d_uu)
        return H


class QuadraticOracle(BaseSmoothOracle):
    """f(x) = 1/2 x^T A x - b^T x with symmetric A."""

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError("A should be a symmetric matrix.")
        self.A = A
        self.b = np.asarray(b, dtype=float).ravel()

    def func(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A


class LogCoshL2Oracle(BaseSmoothOracle):
    """
    Smooth loss: mean(logcosh(Ax - b)) + (regcoef/2)||x||^2.
    For constrained L1-ball experiments set regcoef=0 so f is only the smooth data term.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef=0.0):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = np.asarray(b, dtype=float).ravel()
        self.regcoef = float(regcoef)
        self._m = max(int(self.b.size), 1)

    def func(self, x):
        x = np.asarray(x, dtype=float).ravel()
        r = self.matvec_Ax(x) - self.b
        lc = np.logaddexp(r, -r) - _LOG2
        return np.mean(lc) + (self.regcoef / 2.0) * np.dot(x, x)

    def grad(self, x):
        x = np.asarray(x, dtype=float).ravel()
        r = self.matvec_Ax(x) - self.b
        t = np.tanh(r)
        return (1.0 / self._m) * self.matvec_ATx(t) + self.regcoef * x

    def hess(self, x):
        x = np.asarray(x, dtype=float).ravel()
        r = self.matvec_Ax(x) - self.b
        sech2 = 1.0 - np.tanh(r) ** 2
        s = sech2 / self._m
        return self.matmat_ATsA(s) + self.regcoef * np.eye(len(x))


class LogisticLossOracle(BaseSmoothOracle):
    """
    Mean logistic loss: (1/m) * sum log(1 + exp(-y_i * (Ax)_i)), y_i in {-1, +1}.
    """

    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, y, l2_reg=0.0):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.y = np.asarray(y, dtype=float).ravel()
        self.l2_reg = float(l2_reg)
        self._m = max(int(self.y.size), 1)

    def func(self, x):
        x = np.asarray(x, dtype=float).ravel()
        z = self.y * self.matvec_Ax(x)
        return np.mean(np.logaddexp(0.0, -z)) + (self.l2_reg / 2.0) * np.dot(x, x)

    def grad(self, x):
        x = np.asarray(x, dtype=float).ravel()
        z = self.y * self.matvec_Ax(x)
        sig = expit(-z)
        w = -(self.y * sig) / self._m
        return self.matvec_ATx(w) + self.l2_reg * x

    def hess(self, x):
        x = np.asarray(x, dtype=float).ravel()
        z = self.y * self.matvec_Ax(x)
        p = expit(-z)
        s = p * (1.0 - p) / self._m
        return self.matmat_ATsA(s) + self.l2_reg * np.eye(len(x))


def _dense_matvec_factory(A):
    A = np.asarray(A, dtype=float)

    def Ax(v):
        return A.dot(np.asarray(v, dtype=float).ravel())

    def ATx(u):
        return A.T.dot(np.asarray(u, dtype=float).ravel())

    def ATsA(s):
        su = np.asarray(s, dtype=float).ravel()
        return A.T @ (su.reshape(-1, 1) * A)

    return Ax, ATx, ATsA


class CompositeNonsmoothOracle(BaseNonsmoothConvexOracle):
    """F(x) = f(x) + lambda * ||x||_1 with f smooth convex."""

    def __init__(self, smooth_f, lam_l1):
        self._f = smooth_f
        self.lam_l1 = float(lam_l1)

    def func(self, x):
        x = np.asarray(x, dtype=float).ravel()
        return float(self._f.func(x) + self.lam_l1 * np.linalg.norm(x, 1))

    def subgrad(self, x):
        x = np.asarray(x, dtype=float).ravel()
        g = self._f.grad(x)
        v = np.zeros_like(x)
        for i in range(x.size):
            if abs(x[i]) > 1e-14:
                v[i] = np.sign(x[i])
            else:
                gi = g[i]
                if abs(self.lam_l1) < 1e-30:
                    v[i] = 0.0
                else:
                    v[i] = float(np.clip(-gi / self.lam_l1, -1.0, 1.0))
        return g + self.lam_l1 * v


class RegressionSmoothOracle(LogCoshL2Oracle):
    """Robust regression smooth term (log-cosh residual)."""

    pass


class ClassificationSmoothOracle(LogisticLossOracle):
    """Classification smooth term (logistic loss)."""

    pass


class RegressionNonsmoothOracle(CompositeNonsmoothOracle):
    def __init__(self, A, b, regcoef, l2_reg=0.0):
        Ax, ATx, ATsA = _dense_matvec_factory(A)
        f = LogCoshL2Oracle(Ax, ATx, ATsA, b, regcoef=l2_reg)
        super().__init__(f, regcoef)


class ClassificationNonsmoothOracle(CompositeNonsmoothOracle):
    def __init__(self, A, y, regcoef, l2_reg=0.0):
        Ax, ATx, ATsA = _dense_matvec_factory(A)
        f = LogisticLossOracle(Ax, ATx, ATsA, y, l2_reg=l2_reg)
        super().__init__(f, regcoef)


class RegressionProxOracle(BaseCompositeOracle):
    def __init__(self, A, b, regcoef, l2_reg=0.0):
        Ax, ATx, ATsA = _dense_matvec_factory(A)
        f = LogCoshL2Oracle(Ax, ATx, ATsA, b, regcoef=l2_reg)
        h = L1RegOracle(regcoef)
        super().__init__(f, h)


class ClassificationProxOracle(BaseCompositeOracle):
    def __init__(self, A, y, regcoef, l2_reg=0.0):
        Ax, ATx, ATsA = _dense_matvec_factory(A)
        f = LogisticLossOracle(Ax, ATx, ATsA, y, l2_reg=l2_reg)
        h = L1RegOracle(regcoef)
        super().__init__(f, h)
