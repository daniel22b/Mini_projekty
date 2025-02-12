from typing import List

Vector = List[float]
wzors_waga_wiek =[170,
                  70,
                  40]

oceny = [95,
         80,
         75,
         65]

def add(v: Vector, w: Vector)-> Vector:
    assert len(v) == len(w), "Wektory musza byc tej sameej dlugosci"
    return [v_i + w_i for v_i, w_i in zip(v,w)]

assert add([1,2,3], [4,5,6]) == [5,7,9]

def substract(v: Vector, w: Vector)-> Vector:
    assert len(v) == len(w), "Wektory musza byc tej sameej dlugosci"
    return [v_i - w_i for v_i, w_i in zip(v,w)]

def vector_sum(vectors: List[Vector])-> Vector:
    assert vectors,"Brak wektorow"

    num_elements = len(vectors[0])

    assert all(len(v) == num_elements for v in vectors), "rozne dlugosci"

    return [sum(vector[i] for vector in vectors)
            for i in range(num_elements)]

    assert vector_sum([[1,2],[3,4],[5,6],[7,8]]) == [16,20]

def scallar_multiply(c: float, v: Vector) -> Vector:
    return [c* v_i for v_i in v]

assert scallar_multiply(2,[1,2,3]) == [2,4,6]

def vector_mean(vecotrs:List[Vector]) -> Vector:
    n =len(vecotrs)
    return scallar_multiply(1/n, vector_sum(vecotrs))


def dot(v: Vector, w: Vector)->float:
    assert len(w) == len(v), "wektroy musza byc tej samej dlugosci"
    return sum([v_i * w_i for v_i, w_i in zip(v,w)])

assert dot([1,2],[3,4]) == 11

def sum_of_squers(v: Vector) -> float:
    return dot(v,v)
assert sum_of_squers([1,2,3]) == 14, "Bląd sumy kwadratow"

import math


def squardte_distance(v:Vector, w: Vector) -> float:
    return sum_of_squers(substract(v, w))

def distance(v: Vector, w:Vector) ->float:
    return math.sqrt(squardte_distance(v,w))

def mignitude(v: Vector) -> float: #pierwiastek z 14
    return math.sqrt(sum_of_squers(v))

def distance_v2(v: Vector, w:Vector) -> float:
    return mignitude(substract(v,w))

def mean(xs: List[float]) -> float:
    return sum(xs)/len(xs)
#_____________________________________________________________________


def data_range(xs: List[float]) ->  float:
    return max(xs) - min(xs)

def de_mean(xs: List[float]) -> List[float]:
    x_bar = mean(xs)
    return [x - x_bar for x in xs]

def variance(xs: List[float])-> float:
    n = len(xs)
    deviation = de_mean(xs)
    return sum_of_squers(deviation)/ (n - 1)

from scratch.linear_algebra import dot 

def covariance(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys), "xs i ys musza byc tej samej dlugosci"
    return dot(de_mean(xs), de_mean(ys))/ (len(xs)-1)
