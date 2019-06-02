import errno
import itertools
import os
import re
import sys

import numpy as np
from ecdsa.numbertheory import factorization
from numba import jit


# Класс Сигнатуры
class Signature():
    def __init__(self, sign, primes):
        self.sign = sign  # list
        self.primes = primes  # list


# преобразование
@jit
def n_to_s_d(n):
    d = n - 1
    power = 0
    while d % 2 == 0:
        d //= 2
        power += 1
    return power, d


# возведение в степень по модулю
@jit
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


# чтение из файла 1млн простых чисел и прочих
def readfile(path):
    with open(path, 'r') as f:
        arr = np.array(f.read().split(), dtype=np.uint32)
    f.close()
    return arr


# в основном парсинг n_list'ов и equal_signs
def parsefile(path):
    with open(path, 'r') as f:
        s = f.read()
        signs, primes = None, None

        if 'equal' in path:
            signs = re.findall(r'(\[\d+.*\])\s{5}', s)
            primes = re.findall(r'\s{5}(\[\d+.*\])', s)
        elif 'n_list' in path:
            primes = re.findall(r'(\[\d+.*\])\s{4}', s)
            signs = re.findall(r'\s{4}(\[\d+.*\])', s)

        signatures = []
        for i in range(len(signs)):
            item_s = list(map(int, re.sub(r'[\W]', '', signs[i])))
            item_p = list(map(int, re.split(r' ', re.sub(r'[\[\],]', '', primes[i]))))
            signatures.append(Signature(item_s, item_p))
        return signatures


# запись в файл
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


# очистка файла
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


# распределение простых чисел по классовым остаткам
def Primes_modulo(primes):
    clearfile("primes/4k+1.txt")
    clearfile("primes/4k+3.txt")
    clearfile("primes/8k+1.txt")
    clearfile("primes/8k+5.txt")
    clearfile("primes/else.txt")

    p_1mod4, p_3mod4, p_1mod8, p_5mod8, p_else = [], [], [], [], []
    for p in primes:
        print(p)
        if p % 8 == 5:
            p_5mod8.append(p)
        if p % 8 == 1:
            p_1mod8.append(p)
        if p % 4 == 1:
            p_1mod4.append(p)
        if p % 4 == 3:
            p_3mod4.append(p)
        if p % 8 != 5 and p % 8 != 1 and p % 4 != 1 and p % 4 != 3:
            p_else.append(p)

    p__1mod4 = ''.join(str(l) + ' ' + '\n' * (n % 8 == 7) for n, l in enumerate(p_1mod4))
    p__3mod4 = ''.join(str(l) + ' ' + '\n' * (n % 8 == 7) for n, l in enumerate(p_3mod4))
    p__1mod8 = ''.join(str(l) + ' ' + '\n' * (n % 8 == 7) for n, l in enumerate(p_1mod8))
    p__5mod8 = ''.join(str(l) + ' ' + '\n' * (n % 8 == 7) for n, l in enumerate(p_5mod8))
    p__else = ''.join(str(l) + ' ' + '\n' * (n % 8 == 7) for n, l in enumerate(p_else))

    writefile("primes/4k+1.txt", p__1mod4)
    writefile("primes/4k+3.txt", p__3mod4)
    writefile("primes/8k+1.txt", p__1mod8)
    writefile("primes/8k+5.txt", p__5mod8)
    writefile("primes/else.txt", p__else)

    return len(np.array(p_1mod4)), len(np.array(p_3mod4)), len(np.array(p_1mod8)), len(np.array(p_5mod8)), len(
        np.array(p_else))


if __name__ == "__main__":
    print(sys.version)
    # t = 3
    a_base = [2, 3]
    # B = 10 ** 8
    print(parsefile(f"lib/equal/{a_base}/equal_signs.txt"))
    # s = parsefile(f"res/jnd/{t}/{a_base}/n_list_{B//100}_{B}.txt")
