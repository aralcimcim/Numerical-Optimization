import numpy as np


def rosenbrock_function():
    def function(x):
        return 100 * (x[1] - x[0] ** 2) ** 2 + (1 - x[0]) ** 2

    def gradient(x):
        return np.array([2 * (200 * x[0] ** 3 - 200 * x[0] * x[1] + x[0] - 1),
                         200 * (x[1] - x[0] ** 2)])

    def hessian(x):
        return np.array([[-400 * (x[1] - x[0] ** 2) + 800 * x[0] ** 2 + 2, - 400 * x[0]],
                         [-400 * x[0], 200]])

    def minima():
        return [np.array([1, 1])]

    return function, gradient, hessian, minima()


def f():
    def function(x):
        return 150 * (x[0] * x[1]) ** 2 + (0.5 * x[0] + 2 * x[1] - 2) ** 2

    def gradient(x):
        return np.array([x[0] * (300 * x[1] ** 2 + 0.5) + 2 * x[1] - 2,
                         300 * x[0] ** 2 * x[1] + 2 * x[0] + 8 * x[1] - 8])

    def hessian(x):
        return np.array([[300 * x[1] ** 2 + 0.5, 600 * x[0] * x[1] + 2],
                         [600 * x[0] * x[1] + 2, 300 * x[0] ** 2 + 8]])

    def minima():
        return [np.array([0, 1]), np.array([4, 0])]

    return function, gradient, hessian, minima()


def backtracking_line_search(function_f, gradient_f, x_k, p_k, max_iters=1e4, rho=0.9, c=0.01):
    alpha_bar = 1  # same as in the book p.37
    # rho between 0, 1
    # c between 0, 1

    n_iters = 0

    alpha = alpha_bar
    while function_f(x_k + alpha * p_k) > function_f(x_k) + c * alpha * gradient_f(x_k) @ p_k:
        if n_iters > max_iters:
            return alpha
        alpha = rho * alpha
        n_iters += 1
    return alpha


def closest_index(number, number_list):
    """
    Returns the index of the element in a list that is closest to the given number.
    Used to determine the closest solution to the result.
    """
    closest_i = None
    min_difference = float('inf')

    for i, num in enumerate(number_list):
        difference = np.linalg.norm(number - num)
        if difference < min_difference:
            min_difference = difference
            closest_i = i

    return closest_i
