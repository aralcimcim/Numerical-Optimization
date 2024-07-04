from utils import backtracking_line_search
import numpy as np


def fletcher_reeves(f, grad_f, x0, tol=1e-6, max_iter=1e4):
    x = x0
    r = -grad_f(x)
    p = r
    k = 0
    while np.linalg.norm(r) > tol and k < max_iter:
        alpha = backtracking_line_search(f, grad_f, x, p)
        x = x + alpha * p
        r_new = -grad_f(x)
        beta = np.dot(r_new, r_new) / np.dot(r, r)
        p = r_new + beta * p
        r = r_new
        k += 1
    return x, k


# page 121-123
def polak_ribiere(function,
                  gradient,
                  starting_point,
                  stopping_criterion=1e-6,
                  max_iterations=1e4):
    x_k = starting_point
    r_k = -gradient(x_k)
    p_k = r_k

    n_iters = 0
    while np.linalg.norm(r_k) > stopping_criterion and n_iters < max_iterations:
        alpha_k = backtracking_line_search(function, gradient, x_k, p_k, c=0.001, rho=0.4)
        x_k = x_k + alpha_k * p_k
        r_k_plus_1 = -gradient(x_k)
        beta_k = - r_k_plus_1.T @ (-r_k_plus_1 + r_k) / (np.linalg.norm(r_k) ** 2)
        p_k = r_k_plus_1 + beta_k * p_k
        r_k = r_k_plus_1
        n_iters += 1

    return x_k, n_iters
