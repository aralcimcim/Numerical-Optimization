import numpy as np


def approximate_gradient(function_f, epsilon=1e-6):
    def f(x):
        # Use central difference formula, page 196
        gradient = np.zeros_like(x)
        for i in range(len(gradient)):
            x_new_1 = x.copy()
            x_new_2 = x.copy()
            x_new_1[i] += epsilon
            x_new_2[i] -= epsilon
            gradient[i] = (function_f(x_new_1) - function_f(x_new_2)) / (2 * epsilon)
        return gradient

    return f


def approximate_hessian(function_f, epsilon=1e-6):
    def f(x):
        hessian = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                if i <= j:
                    x_ij = x.copy()
                    x_i = x.copy()
                    x_j = x.copy()

                    x_ij[i] += epsilon
                    x_ij[j] += epsilon

                    x_i[i] += epsilon
                    x_j[j] += epsilon

                    grad_ij = (function_f(x_ij) - function_f(x_i) - function_f(x_j) + function_f(x)) / (epsilon ** 2)
                    hessian[i, j] = grad_ij
                    hessian[j, i] = grad_ij
        return hessian

    return f
