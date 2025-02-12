import random
from typing import TypeVar, List, Tuple
import numpy as np


X = TypeVar('X')

def split_data(data:List[X], prob: float)-> Tuple[List[X], List[X]]:
    data = data[:]
    random.shuffle(data)
    cut = int(len(data) * prob)
    return data[:cut], data[cut:]

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

assert len(train + test) == 1000, "ERROR"


Y = TypeVar('Y')

def train_test_split(xs: List[X], ys:List[Y], test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
    idxs = [i for i in range(len(xs))]
    train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

    return ([xs[i] for i in train_idxs],
            [xs[i] for i in test_idxs],
            [ys[i] for i in train_idxs],
            [ys[i] for i in test_idxs])

from sklearn.model_selection import train_test_split

data = np.arange(1000)

labels = np.random.randint(0,2, size=1000)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, random_state = 42)


print("Train size:", len(x_train))
print("Test size:", len(x_test))