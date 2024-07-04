import numpy as np
from utils import backtracking_line_search


def cholesky_with_added_multiple_of_identity(hessian_matrix):
    # page 52: "typical value is 10^-3"
    beta = 0.001
    if min(np.diag(hessian_matrix)) > 0:
        tau = 0
    else:
        tau = -min(np.diag(hessian_matrix)) + beta

    while True:
        try:
            L = np.linalg.cholesky(hessian_matrix + tau * np.eye(len(hessian_matrix)))
            return L @ L.conj().T
        except np.linalg.LinAlgError:
            tau = max(2 * tau, beta)


def newton_method(starting_point,
                  function_f,
                  gradient_f,
                  hessian_f,
                  stopping_criterion=1e-6,
                  max_iterations=1e4,
                  hessian_modification=False):
    x_k = starting_point
    n_iter = 0
    while np.linalg.norm(gradient_f(x_k)) > stopping_criterion:
        if n_iter >= max_iterations:
            return x_k, n_iter
        # hess_f p_k = -gradient_f --> p_k = - hess_f ^-1 gradient_f
        if hessian_modification:
            x = (cholesky_with_added_multiple_of_identity(hessian_f(x_k).astype(float)))
        else:
            x = hessian_f(x_k).astype(float)
        y = gradient_f(x_k).astype(float)
        p_k = np.linalg.solve(x, -y)

        alpha_k = backtracking_line_search(function_f, gradient_f, x_k, p_k)

        x_k = x_k + alpha_k * p_k
        n_iter += 1

    return x_k, n_iter
