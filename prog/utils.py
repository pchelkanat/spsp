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


def powmod(a, x, p):
    x_bin = list(bin(x)[2:])
    a, p = np.uint32(a), np.uint32(p)
    # print(type(a), type(p))
    mod = a
    for i in range(1, len(x_bin), 1):
        if int(x_bin[i]) == 0:
            mod = mod ** 2 % p
        elif int(x_bin[i]) == 1:
            mod = a * mod ** 2 % p
    return mod


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
# print(powmod(5, 97651 - 1, 97651))
