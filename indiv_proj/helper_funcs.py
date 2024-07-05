import numpy as np
from scipy.optimize import approx_fprime

def backtracking_line_search(func, grad, x_k, p, rho=0.5, c=1e-4, alpha=0.01, method='steepest descent', hessian=None):
    grad_k = grad(x_k)
    if method == 'steepest descent':
        descent = np.dot(grad_k, p(x_k, grad, method=method))
    elif method == 'newton':
        descent = np.dot(grad_k, p(x_k, grad, method=method, hessian=hessian))
    while func(x_k + alpha*p(x_k, grad, method=method, hessian=hessian)) > func(x_k) + c*alpha*descent:
        alpha = rho*alpha
        print(f'alpha = {alpha}')
    alpha_k = alpha
    return alpha_k

def p(x, grad, method='steepest descent', hessian=None, regularizer=1e-4):
    grad_val = grad(x)
    if method == 'steepest descent':
        nrm = np.linalg.norm(grad_val)
        if nrm == 0:
            return np.zeros_like(x)
        else:
            return -1*grad_val / nrm
    elif method == 'newton':
        hessian_val = hessian(x)
        if np.isscalar(hessian_val):
            if hessian_val == 0:
                return np.zeros_like(x)
            else:
                return -grad_val / hessian_val
        else:
            if np.linalg.cond(hessian_val) < 1 / regularizer:
                return -np.linalg.solve(hessian_val, grad_val)
            else:
                hessian_val += regularizer*np.eye(len(x))
                return -np.linalg.solve(hessian_val, grad_val)

def descent(func, grad, p, x_0, alpha, num_iter, method, tol=1e-6, hessian=None):
    f = []
    x = []
    x_old = x_0
    f_old = func(x_0)
    x_new = x_old + alpha*p(x_old, grad, method=method, hessian=hessian)
    f_new = func(x_new)

    f.append(f_old)
    f.append(f_new)
    x.append(x_old)
    x.append(x_new)

    for i in range(num_iter):
        # if np.abs(f_new - f_old) < tol:
        #     break
        #print(f'f(x) = {f_new}')

        if np.linalg.norm(grad(x_new)) < tol:
            print(f'Stopping criterion ({tol}) reached at iteration {i+1}')
            break

        f_old = f_new
        x_old = x_new
        alpha = backtracking_line_search(func, grad, x_old, p, alpha=alpha, method=method, hessian=hessian)
        x_new = x_old + alpha*p(x_old, grad, method=method, hessian=hessian)
        f_new = func(x_new)
        f.append(f_new)
        x.append(x_new)
    return x, f

# polynomials for task iv
def polynomial(coeffs, x):
    q1 = np.polyval(coeffs[:5], x[0])
    q2 = np.polyval(coeffs[5:], x[1])
    return q1**2 + q2**2

def gradient_poly(coeffs, x):
    epsilon = np.sqrt(np.finfo(float).eps)
    return approx_fprime(x, lambda x: polynomial(coeffs, x), epsilon)

def hessian_poly(coeffs, x):
    epsilon = np.sqrt(np.finfo(float).eps)
    return approx_fprime(x, lambda x: gradient_poly(coeffs, x), epsilon)