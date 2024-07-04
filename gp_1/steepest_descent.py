import numpy as np


def steepest_descent(starting_point,
                     hilbert_matrix,
                     gradient_f,
                     stopping_criterion=1e-6,
                     max_iterations=1e4):
    x_k = starting_point
    n_iters = 0

    while np.linalg.norm(gradient_f(x_k)) > stopping_criterion:
        if n_iters >= max_iterations:
            return x_k, n_iters
        p_k = gradient_f(x_k)
        alpha_k = (p_k.T @ p_k) / (p_k.T @ hilbert_matrix @ p_k)
        x_k = x_k - alpha_k * p_k
        n_iters += 1

    return x_k, n_iters
