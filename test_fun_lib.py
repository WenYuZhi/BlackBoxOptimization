import numpy as np 
import math

def sphere_fun(x):
    return (x**2).sum()

def schwefel_fun(x):
    y = 1.0
    for i in range(x.size):
        y = y * x[i]
    return np.abs(x).sum() + abs(y)

def schwefel_fun2(x):
    return np.abs(x).max()

def rosenbrock_fun(x):
    return sum([100*(x[i]**2 - x[i+1])**2 + (x[i] - 1)**2 for i in range(x.size - 1)])

def rastrigin_fun(x):
    return sum([x[i]**2 - 10 * math.cos(2*math.pi*x[i]) + 10 for i in range(x.size)])