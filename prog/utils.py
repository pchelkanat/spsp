import errno
import itertools
import os

import numpy as np
from ecdsa.numbertheory import factorization


def n_to_s_d(n):
    d = n - 1
    power = 0
    while d % 2 == 0:
        d //= 2
        power += 1
    return power, d


# проверка нечетного n на простые неповторяющиеся множители
def check_n(n):
    arr = np.array(factorization(n))  # возвращает (число, его степень)
    temp = arr.T[1]
    if temp.prod() == 1:  # проверка: все ли степени равны 1
        return True, arr.T[0]
    else:
        return False


# чтение из файла 1млн простых чисел
def readfile(path):
    with open(path, 'r') as f:
        arr = np.array(f.read().split(), dtype=np.int32)
    f.close()
    return arr


def writefile(path, item):
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

    with open(path, 'a') as f:
        f.write(item)
    f.close()


def clearfile(path):
    if not os.path.exists(path):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise
    with open(path, 'w') as f:
        f.write("")
    f.close()


# комбинация простых чисел по t штук из полученного ранее списка
def combinations(mylist, t):
    return np.array(list(itertools.combinations(mylist, t)), dtype=np.int32)

# if __name__ == "__main__":
