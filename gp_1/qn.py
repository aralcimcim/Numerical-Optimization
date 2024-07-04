import numpy as np
from utils import backtracking_line_search
import scipy


def bfgs(starting_point,
         function_f,
         gradient_f,
         H0,
         stopping_criterion=1e-6,
         max_iterations=1e4):
    H_k = H0
    x_k = starting_point
    n_iter = 0

    while np.linalg.norm(gradient_f(x_k)) > stopping_criterion:
        if n_iter >= max_iterations:
            return x_k, n_iter

        p_k = -H_k @ gradient_f(x_k)
        alpha_k = backtracking_line_search(function_f, gradient_f, x_k, p_k)
        x_k_plus_1 = x_k + alpha_k * p_k
        s_k = x_k_plus_1 - x_k
        y_k = gradient_f(x_k_plus_1) - gradient_f(x_k)
        # Numbers tend to get so small, that the denominator is approx 0 -> error
        if np.linalg.norm(y_k.T @ s_k) < 1e-50:
            rho_k = 1e10
        else:
            rho_k = 1 / (y_k.T @ s_k)
        H_k = (np.eye(len(H_k)) - rho_k * np.outer(s_k, y_k)) @ H_k @ (
                np.eye(len(H_k)) - rho_k * np.outer(y_k, s_k)) + rho_k * np.outer(s_k, s_k)
        x_k = x_k_plus_1
        n_iter += 1

    return x_k, n_iter


def sr1_line_search(starting_point,
                    function_f,
                    gradient_f,
                    H0,
                    stopping_criterion=1e-6,
                    max_iterations=1e4):
    H_k = H0
    x_k = starting_point
    n_iter = 0

    while np.linalg.norm(gradient_f(x_k)) > stopping_criterion:
        if n_iter >= max_iterations:
            return x_k, n_iter

        p_k = -H_k @ gradient_f(x_k)
        alpha_k = backtracking_line_search(function_f, gradient_f, x_k, p_k)
        x_k_plus_1 = x_k + alpha_k * p_k
        s_k = x_k_plus_1 - x_k
        y_k = gradient_f(x_k_plus_1) - gradient_f(x_k)

        # Numbers tend to get so small, that the denominator is approx 0 -> error
        if np.linalg.norm((s_k - H_k @ y_k) @ y_k) < 1e-50:
            # don't update at all
            # H_k = H_k
            return x_k, n_iter
        else:
            temp = 1 / ((s_k - H_k @ y_k) @ y_k)
            H_k = H_k + np.outer(s_k - H_k @ y_k, s_k - H_k @ y_k) * temp
        x_k = x_k_plus_1
        n_iter += 1

    return x_k, n_iter


# Algorithm 4.1
def get_trust_region_radius(starting_point, function_f, gradient_f, B_k, max_iterations=100):
    delta_head = 100
    x_k = starting_point
    eta = 0.15  # in [0, 1/4)
    delta_k = 1

    for k in range(max_iterations):
        # Obtain p_k by solving (4.3)
        def m_k(p):
            return function_f(x_k) + gradient_f(x_k).T @ p + 0.5 * p.T @ B_k @ p

        def constraint(p):
            return gradient_f(x_k) - np.linalg.norm(p)

        cons = {"type": "ineq", "fun": constraint}
        p_k = scipy.optimize.minimize(m_k, np.zeros_like(x_k), constraints=cons).x

        # Evaluate rho_k from (4.4)
        rho_k = (function_f(x_k) - function_f(x_k + p_k)) / (m_k(np.zeros(2)) - m_k(p_k))

        if rho_k < 0.25:
            delta_k_plus_1 = 1 / 4 * delta_k
        else:
            if rho_k > 3 / 4 and np.linalg.norm(rho_k) == delta_k:
                delta_k_plus_1 = min(2 * delta_k, delta_head)
            else:
                delta_k_plus_1 = delta_k

        if rho_k > eta:
            x_k_plus_1 = x_k + p_k
        else:
            x_k_plus_1 = x_k

        x_k = x_k_plus_1
        delta_k = delta_k_plus_1

    return delta_k


def sr1_trust_region(starting_point,
                     function_f,
                     gradient_f,
                     B0,
                     stopping_criterion=1e-6,
                     max_iterations=1e4):
    delta_k = get_trust_region_radius(starting_point, function_f, gradient_f, B0)  # trust region radius
    eta = 0.0001  # in (0, 0.001)
    eta = 1e-6
    r = 1e-8  # in (0,1) # see page 146

    x_k = starting_point
    B_k = B0
    n_iter = 0

    while np.linalg.norm(gradient_f(x_k)) > stopping_criterion:
        if n_iter >= max_iterations:
            return x_k, n_iter

        # Constraint: "inequality means that it has to be non-negative" --> delta_k - ||s|| >= 0
        def constraint(s):
            return delta_k - np.linalg.norm(s)

        def minimization_subproblem(s):
            return gradient_f(x_k).T @ s + 0.5 * s.T @ B_k @ s

        cons = {"type": "ineq", "fun": constraint}
        s_k = scipy.optimize.minimize(minimization_subproblem, np.zeros_like(x_k), constraints=cons).x

        y_k = gradient_f(x_k + s_k) - gradient_f(x_k)
        ared = function_f(x_k) - function_f(x_k + s_k)  # actual reduction
        pred = -(gradient_f(x_k) @ s_k + 0.5 * s_k.T @ B_k @ s_k)  # predicted reduction

        # Avoid division by zero
        if np.linalg.norm(pred) < 1e-50:
            pred = 1e-10

        if ared / pred > eta:
            x_k_plus_1 = x_k + s_k
        else:
            x_k_plus_1 = x_k

        if ared / pred > 0.75:
            if np.linalg.norm(s_k) <= 0.8 * delta_k:
                delta_k_plus_1 = delta_k
            else:
                delta_k_plus_1 = 2 * delta_k
        elif 0.1 <= ared / pred <= 0.75:
            delta_k_plus_1 = delta_k
        else:
            delta_k_plus_1 = 0.5 * delta_k

        if np.linalg.norm((y_k - B_k @ s_k).T @ s_k) >= 1e-50:
            # (6.26)
            if np.linalg.norm(s_k.T @ (y_k - B_k @ s_k)) >= r * np.linalg.norm(s_k) * np.linalg.norm(y_k - B_k @ s_k):
                B_k_plus_1 = B_k + np.outer(y_k - B_k @ s_k, y_k - B_k @ s_k) / ((y_k - B_k @ s_k).T @ s_k)  # (6.24)
            else:
                B_k_plus_1 = B_k
        else:
            return x_k, n_iter

        n_iter += 1
        x_k = x_k_plus_1
        delta_k = delta_k_plus_1
        B_k = B_k_plus_1

    return x_k, n_iter
