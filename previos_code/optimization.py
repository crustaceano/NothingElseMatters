import numpy as np
from numpy.linalg import LinAlgError
import scipy
from scipy.linalg import cho_factor, cho_solve, eigh
import time
from collections import defaultdict
import copy
from scipy.optimize._linesearch import scalar_search_wolfe2


def stop_criterion(oracle, x_k, x_0, tolerance):
    return np.linalg.norm(oracle(x_k).grad()) ** 2 <= tolerance * np.linalg.norm(oracle(x_0).grad()) ** 2

def update_history(oracle, x_k, history, start_time, trace):
    if trace:
        history['time'].append(time.perf_counter() - start_time)
        history['func'].append(oracle.func(x_k))
        history['grad_norm'].append(np.linalg.norm(oracle(x_k).grad()))
        if x_k.size <= 2:
            history['x'].append(x_k.copy())


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule 
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c
        if previous_alpha is not None:
            alpha = previous_alpha
        else:
            alpha = self.alpha_0
        if self._method == 'Armijo':
            return self.__armijo_search(oracle, x_k, d_k, alpha)
        elif self._method == 'Wolfe':
            return self.__wolfe_search(oracle, x_k, d_k, alpha)
        else:
            raise ValueError('Unknown method {}'.format(self._method))

    def __armijo_search(self, oracle, x_k, d_k, alpha_0):
        alpha = alpha_0
        phi_0 = oracle.func_directional(x_k, d_k, 0)
        grad_phi_0 = oracle.grad_directional(x_k, d_k, 0)

        while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * grad_phi_0:
            alpha /= 2
            if alpha < 1e-8:
                return None
        
        return alpha
    
    def __wolfe_search(self, oracle, x_k, d_k, alpha_0):
        phi = lambda alpha: oracle.func_directional(x_k, d_k, alpha)
        derphi = lambda alpha: oracle.grad_directional(x_k, d_k, alpha)

        alpha, *_ = scalar_search_wolfe2(phi, derphi, c1=self.c1, c2=self.c2)
        if alpha is None:
            return self.__armijo_search(oracle, x_k, d_k, alpha_0)
        return alpha

def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    start_time = time.perf_counter()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    update_history(oracle, x_k, history, start_time, trace)
    is_success = False
    previous_alpha = None

    for k in range(max_iter):
        try:
            direction = -oracle(x_k).grad()
            step_size = line_search_tool.line_search(
                oracle, x_k, direction, previous_alpha
            )
            previous_alpha = step_size
            if (np.any(np.isnan(direction)) or np.any(np.isinf(direction)) or 
                step_size is None or np.isnan(step_size) or np.isinf(step_size)):
                return x_k, "computational_error", history
            x_k += step_size * direction
            if (np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)) or
                oracle(x_k).func() is None or
                np.isnan(oracle(x_k).func()) or
                np.isinf(oracle(x_k).func())):
                return x_k, "computational_error", history
        except Exception:
            return x_k, "computational_error", history

        update_history(oracle, x_k, history, start_time, trace)
        if stop_criterion(oracle, x_k, x_0, tolerance):
            is_success = True
            break

    if is_success:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
            - history['alpha'] : list of line-search step sizes used at each iteration (one entry per successful step)

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    start_time = time.perf_counter()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    update_history(oracle, x_k, history, start_time, trace)
    is_success = False
    previous_alpha = None

    for k in range(max_iter):
        try:
            o = oracle(x_k)
            g = o.grad()
            H = o.hess()
            c, low = cho_factor(H, lower=True)
            d_k = cho_solve((c, low), -g)
        except LinAlgError:
            return x_k, 'newton_direction_error', history
        except Exception:
            return x_k, 'computational_error', history

        try:
            step_size = line_search_tool.line_search(
                oracle, x_k, d_k, previous_alpha
            )
            if (np.any(np.isnan(d_k)) or np.any(np.isinf(d_k)) or
                step_size is None or np.isnan(step_size) or np.isinf(step_size)):
                return x_k, 'computational_error', history
            previous_alpha = step_size
            x_k += step_size * d_k
            if (np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)) or
                oracle(x_k).func() is None or
                np.isnan(oracle(x_k).func()) or
                np.isinf(oracle(x_k).func())):
                return x_k, 'computational_error', history
        except Exception:
            return x_k, 'computational_error', history

        if trace:
            history['alpha'].append(float(step_size))

        update_history(oracle, x_k, history, start_time, trace)
        if stop_criterion(oracle, x_k, x_0, tolerance):
            is_success = True
            break

    if is_success:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history
