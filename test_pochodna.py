# from scratch.linear_algebra import Vector, dot, scalar_multiply, add
from typing import Callable, List
Vector = List[float]
def partial_difference_quotient(f:Callable[[Vector], float],
                                v:Vector,
                                i: int,
                                h:float)->float:
    
    w = [v_j +(h if j == i else 0)
        for j, v_j in enumerate(v)]
    
    return (f(w) - f(v)) / h

def estimate_difference_quotient(f:Callable[[Vector], float],
                                v:Vector,
                                h:float)->Vector:
    
    return [partial_difference_quotient(f,v,i,h) for i in range(len(v))]

def f(v:Vector) -> float:
    x,y,z = v
    return x**2 + y**2 + z**2

v =[1,2,3]
print(partial_difference_quotient(f, v, 0, 0.001))
print(estimate_difference_quotient(f,v,0.001))


def square(x):
    return x **2

def apply_func(f,x):
    return f(x)

result = apply_func(square, 3)
print(result)

def partial(f: Callable[[Vector], float],
                v: Vector,
                i : int,
                h: float)->float:
    
    w = [v_j + (h if j==i else 0) for j,v_j in enumerate(v)]
    return (f(w) - f(v))/h


inputs = [(x,x**2 +5) for x in range(-10,20)]
import numpy as np

def grad_step(v:Vector, g: Vector, step_size: float)-> Vector:
    step = np.dot(step_size,g)
    return np.add(step, v)

def gradient(v:Vector)-> Vector:
    return [2*v_i for v_i in v]

v = [1,4,2,6]
for epoch in range(100):
    grad = gradient(v)
    v = grad_step(v,grad, -0.001)
    print(epoch, v)