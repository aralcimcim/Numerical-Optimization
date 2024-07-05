import numpy as np

def func1(x):
    return x**3 - 3*x**2 + 2*x - 1

def func2(x):
    return x**4 - 6*x**2 + 9

def func3(x):
    return x**5 - 10*x**3 + 25*x

def func4(x):
    return x**4 - 12*x**2 + 36

def func5(x):
    return x**3 - 6*x**2 + 9

def grad1(x):
    return 3*x**2 - 6*x + 2

def grad2(x):
    return 4*x**3 - 12*x

def grad3(x):
    return 5*x**4 - 30*x**2 + 25

def grad4(x):
    return 4*x**3 - 24*x

def grad5(x):
    return 3*x**2 - 12*x

def hessian1(x):
    return 6*x - 6

def hessian2(x):
    return 12*x**2 - 12

def hessian3(x):
    return 20*x**3 - 60*x

def hessian4(x):
    return 12*x**2 - 24

def hessian5(x):
    return 6*x - 12
