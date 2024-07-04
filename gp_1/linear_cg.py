import numpy as np


def linear_cg(starting_point,
              hilbert_matrix,
              stopping_criterion=1e-6,
              max_iterations=1e4):
    x_k = starting_point
    b = np.ones((len(hilbert_matrix),), dtype=np.longfloat)
    r_k = hilbert_matrix @ x_k - b
    p_k = -r_k

    n_iters = 0

    while np.linalg.norm(r_k) > stopping_criterion:
        if n_iters >= max_iterations:
            return x_k, n_iters

        alpha_k = (r_k.T @ r_k) / (p_k.T @ hilbert_matrix @ p_k)
        x_k = x_k + alpha_k * p_k
        r_k_plus_1 = r_k + alpha_k * hilbert_matrix @ p_k
        beta_k = (r_k_plus_1.T @ r_k_plus_1) / (r_k.T @ r_k)
        p_k = -r_k_plus_1 + beta_k * p_k
        r_k = r_k_plus_1
        n_iters += 1

    return x_k, n_iters
