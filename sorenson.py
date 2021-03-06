from gmpy2 import gcd, root

import ecdsa.numbertheory as numth
import numpy as np

from jaeschke import Val, Ord, Sign
from utils import readfile, n_to_s_d

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

Q11 = 3825123056546413051
primes = readfile("primes/primes_1m.txt")


class Signature():
    def __init__(self, sign, primes_lmd):
        self.sign = sign  # list
        self.primes = primes  # list

    def insert(self, sign_p, prime, lmd):
        if sign_p in signs:
            temp = sign_p.primes
            if isinstance(temp, list):
                temp.append([prime, lmd])
            else:
                temp.append([[prime, lmd]])
            sign_p.primes = sorted(temp)

    def fetch(self, sign_p):
        if sign_p in signs:
            return sign_p.primes


# Псевдопростое по базе
def psp(b, k, pt):
    if b ** (k - 1) % pt == 1:
        return True
    else:
        return False


# Псевдопростое по базам
def check_psp(a_base, k, pt):
    a_list = []
    for a in a_base:
        a_list.append(a ** (k - 1) - 1)
    gcd_of_list = numth.gcd(a_list)
    if gcd_of_list % pt == 0:
        return True
    else:
        return False


def h(b, k):
    d, u = n_to_s_d(k)
    c = Val(2, (Ord(k, b)))
    if c == 0:
        return b ** u - 1
    elif c > 0:
        return b ** (u * 2 ** (c - 1)) + 1


def hashes(a_base, k):
    h_list = []
    for a in a_base:
        h_list.append(h(a, k))
    return h_list


def Rule_out_k(a_base, k):
    ind_min = np.argmin(np.array(hashes(a_base, k)))
    b = a_base[ind_min]
    m = len(a_base)
    i = None
    if b == a_base[0]:
        i = 2
    else:
        i = 1
    hash = h(b, k)
    x = h(a_base[i], k) % hash
    x = gcd(hash, x)
    while x > k and i < m:
        i += 1
        if a_base[i] == b:
            i += 1
        y = h(a_base[i], k) % x
        x = gcd(x, y)
    if x < k:
        print("исключаем k")
        return False
    else:
        factors = numth.factorization(x)
        for i in range(len(factors)):
            pt = factors[i][0]
            if pt > k and check_psp(a_base, k, pt):
                return pt


def Lambda_p(a_base, p):
    ord_base = []
    for a in a_base:
        ord_base.append(Ord(p, a))
    lmd_p = numth.lcm(ord_base)
    return lmd_p


def Lambda_list(a_base, primes):
    lmd_list = []
    for p in primes:
        lmd_list.append(Lambda_p(a_base, p))
    lmd = numth.lcm(lmd_list)
    return lmd


def Sieving(a_base, k_list, B):
    lmd_k = Lambda_list(a_base, k_list)
    k = np.prod(k_list)
    m = len(a_base)
    w = 1
    for i in range(2, m, 1):
        if k * lmd_k * w < B / 1000:
            if lmd_k % a_base[i] != 0:
                w *= a_base[i]
    if lmd_k % 4 == 0:
        lmd = lmd_k
    else:
        lmd = lmd_k / 2
        w = 8 * w
    sign = Sign(a_base, k_list[0])


def Gen_k(a_base, t, B, X):
    m = len(a_base)
    T = Hash_Table()
    a = bases[m]
    for p in primes:
        if p <= int(root(B / (a ** (t - 2)), 2)):
            factors = numth.factorization(p - 1)

            nums = []
            for i in range(len(factors)):
                nums.append(Lambda_p(a_base, factors[i][0]))
            lmd_p = Lambda_list(a_base, primes)
            s = T.fetch(Sign(a_base, p))
            k = np.prod(s) / s[-1] * p
            k_list = s[:-1] + [p]
            if k <= X:
                pt = Rule_out_k(a_base, k)
            else:
                pt = Sieving(a_base, k_list, B)
            if p <= int(root(B / (a ** (t - 3)), 3)):
                T.insert(Sign(a_base, p), p, lmd_p)

    return


if __name__ == "__main__":
    for p in primes[9:20]:
        signs = Hash_Table()
