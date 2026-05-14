"""Wrap smooth / composite oracles to count operator evaluations."""


class CountedSmoothOracle(object):
    """Wraps BaseSmoothOracle-like object."""

    def __init__(self, inner):
        self._inner = inner
        self.n_func = 0
        self.n_grad = 0
        self.n_hess = 0

    def func(self, x):
        self.n_func += 1
        return self._inner.func(x)

    def grad(self, x):
        self.n_grad += 1
        return self._inner.grad(x)

    def hess(self, x):
        self.n_hess += 1
        return self._inner.hess(x)

    def total_calls(self):
        return self.n_func + self.n_grad + self.n_hess


class CountedCompositeOracle(object):
    """Wraps composite oracle (func, grad, prox)."""

    def __init__(self, inner):
        self._inner = inner
        self.n_func = 0
        self.n_grad = 0
        self.n_prox = 0

    def func(self, x):
        self.n_func += 1
        return self._inner.func(x)

    def grad(self, x):
        self.n_grad += 1
        return self._inner.grad(x)

    def prox(self, x, alpha):
        self.n_prox += 1
        return self._inner.prox(x, alpha)

    def total_calls(self):
        return self.n_func + self.n_grad + self.n_prox


class CountedNonsmoothOracle(object):
    def __init__(self, inner):
        self._inner = inner
        self.n_func = 0
        self.n_subgrad = 0

    def func(self, x):
        self.n_func += 1
        return self._inner.func(x)

    def subgrad(self, x):
        self.n_subgrad += 1
        return self._inner.subgrad(x)

    def total_calls(self):
        return self.n_func + self.n_subgrad


class CountedBarrierOracle(object):
    """For z in R^{2n}."""

    def __init__(self, inner):
        self._inner = inner
        self.n_func = 0
        self.n_grad = 0
        self.n_hess = 0

    def _split(self, z):
        return self._inner._split(z)

    def func(self, z):
        self.n_func += 1
        return self._inner.func(z)

    def grad(self, z):
        self.n_grad += 1
        return self._inner.grad(z)

    def hess(self, z):
        self.n_hess += 1
        return self._inner.hess(z)

    def total_calls(self):
        return self.n_func + self.n_grad + self.n_hess
