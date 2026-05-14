import numpy as np
import time
from collections import defaultdict, deque  # Use this for effective implementation of L-BFGS
from utils import get_line_search_tool


def linear_conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    x_k = np.copy(x_0)
    n = b.size
    if max_iter is None:
        max_iter = n

    start_time = time.perf_counter()
    g_k = matvec(x_k) - b
    b_norm = np.linalg.norm(b)
    if b_norm == 0:
        b_norm = 1.0

    def push_history():
        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['residual_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

    push_history()
    if np.linalg.norm(g_k) <= tolerance * b_norm:
        return x_k, 'success', history

    d_k = -g_k
    g_norm_sq = np.dot(g_k, g_k)

    for _ in range(max_iter):
        Ad_k = matvec(d_k)
        denom = np.dot(d_k, Ad_k)
        if denom <= 0:
            return x_k, 'iterations_exceeded', history
        alpha_k = g_norm_sq / denom
        x_k = x_k + alpha_k * d_k
        g_k = g_k + alpha_k * Ad_k

        push_history()
        g_next_norm_sq = np.dot(g_k, g_k)
        if np.sqrt(g_next_norm_sq) <= tolerance * b_norm:
            return x_k, 'success', history

        beta_k = g_next_norm_sq / g_norm_sq
        d_k = -g_k + beta_k * d_k
        g_norm_sq = g_next_norm_sq

    return x_k, 'iterations_exceeded', history

def nonlinear_conjugate_gradients(oracle, x_0, tolerance=1e-4, max_iter=500,
                                  line_search_options=None, display=False, trace=False):
    """
    Nonlinear Conjugate Gradients method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or 'iterations_exceeded'
    history : dictionary of lists or None
        Contains history['func'], history['time'], history['grad_norm'], history['x']
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = time.perf_counter()

    g_k = oracle.grad(x_k)
    gs_norm = np.dot(g_k, g_k)
    if gs_norm == 0:
        gs_norm = 1.0
    d_k = -g_k
    alpha_prev = None

    def push_history():
        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

    push_history()

    for _ in range(max_iter):
        if np.dot(g_k, g_k) <= tolerance * gs_norm:
            return x_k, 'success', history

        if np.dot(g_k, d_k) >= 0:
            d_k = -g_k

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_prev)
        if alpha_k is None or alpha_k <= 0:
            alpha_k = 1e-4
        alpha_prev = alpha_k

        x_next = x_k + alpha_k * d_k
        g_next = oracle.grad(x_next)

        beta_pr = np.dot(g_next, g_next - g_k) / max(np.dot(g_k, g_k), 1e-32)
        if beta_pr < 0:
            beta_pr = 0.0
        d_next = -g_next + beta_pr * d_k
        if np.dot(g_next, d_next) >= 0:
            d_next = -g_next

        x_k = x_next
        g_k = g_next
        d_k = d_next
        push_history()

    return x_k, 'iterations_exceeded', history

def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = time.perf_counter()

    s_history = deque(maxlen=memory_size)
    y_history = deque(maxlen=memory_size)
    rho_history = deque(maxlen=memory_size)
    alpha_prev = None

    g_k = oracle.grad(x_k)
    gs_norm = np.dot(g_k, g_k)
    if gs_norm == 0:
        gs_norm = 1.0

    def push_history():
        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

    def lbfgs_direction(grad):
        if memory_size == 0 or len(s_history) == 0:
            return -grad

        q = np.copy(grad)
        alphas = []
        for s_i, y_i, rho_i in zip(reversed(s_history), reversed(y_history), reversed(rho_history)):
            alpha_i = rho_i * np.dot(s_i, q)
            alphas.append(alpha_i)
            q = q - alpha_i * y_i

        s_last = s_history[-1]
        y_last = y_history[-1]
        gamma = np.dot(s_last, y_last) / max(np.dot(y_last, y_last), 1e-32)
        r = gamma * q

        for i, (s_i, y_i, rho_i) in enumerate(zip(s_history, y_history, rho_history)):
            beta_i = rho_i * np.dot(y_i, r)
            alpha_i = alphas[len(alphas) - 1 - i]
            r = r + s_i * (alpha_i - beta_i)
        return -r

    push_history()
    for _ in range(max_iter):
        if np.dot(g_k, g_k) <= tolerance * gs_norm:
            return x_k, 'success', history

        d_k = lbfgs_direction(g_k)
        if np.dot(g_k, d_k) >= 0:
            d_k = -g_k

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=alpha_prev)
        if alpha_k is None or alpha_k <= 0:
            alpha_k = 1e-4
        alpha_prev = alpha_k

        x_next = x_k + alpha_k * d_k
        g_next = oracle.grad(x_next)
        s_k = x_next - x_k
        y_k = g_next - g_k
        ys = np.dot(y_k, s_k)
        if ys > 1e-12 and memory_size > 0:
            s_history.append(s_k)
            y_history.append(y_k)
            rho_history.append(1.0 / ys)

        x_k = x_next
        g_k = g_next
        push_history()

    return x_k, 'iterations_exceeded', history


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500, 
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    start_time = time.perf_counter()

    g_k = oracle.grad(x_k)
    gs_norm = np.dot(g_k, g_k)
    if gs_norm == 0:
        gs_norm = 1.0

    def push_history():
        if trace:
            history['time'].append(time.perf_counter() - start_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.linalg.norm(g_k))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))

    push_history()

    for _ in range(max_iter):
        g_norm = np.linalg.norm(g_k)
        if g_norm ** 2 <= tolerance * gs_norm:
            return x_k, 'success', history

        eta_k = min(0.5, np.sqrt(g_norm))
        b_cg = -g_k

        d_init = -g_k
        while True:
            hv = lambda v: oracle.hess_vec(x_k, v)
            d_k, _, _ = linear_conjugate_gradients(
                matvec=hv,
                b=b_cg,
                x_0=d_init,
                tolerance=eta_k,
                max_iter=None,
                trace=False,
                display=False
            )
            if np.dot(g_k, d_k) < 0:
                break
            eta_k *= 0.1
            d_init = d_k

        alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        if alpha_k is None or alpha_k <= 0:
            alpha_k = 1e-4

        x_k = x_k + alpha_k * d_k
        g_k = oracle.grad(x_k)
        push_history()

    return x_k, 'iterations_exceeded', history