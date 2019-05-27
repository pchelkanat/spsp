import time
from gmpy2 import root

import ecdsa.numbertheory as numth
import numpy as np

from utils import powmod, readfile, clearfile, Signature, writefile

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

primes = readfile("primes/primes_1m.txt")


def Ord(p, a):
    # p is prime
    # a is integer that gcd(a,p)=1
    # e is the smallest of a**e = 1 mod p
    """
    if math.gcd(p, a) == 1:
        e = bsgs(a, 1, p)
        return e
    """
    return numth.order_mod(a, p)


def Val(p, n):
    # p is prime
    # n is integer
    # e is the greatest power of p in n
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


# вычисление сигнатуры способ 1
def Sign(v, p):
    if p % 4 == 3:
        sgm_v_p = []
        for a in v:
            if numth.jacobi(a, p) == 1:
                sgm_v_p.append(0)
            elif numth.jacobi(a, p) == -1:
                sgm_v_p.append(1)
        return sgm_v_p
    elif p % 4 == 1:
        sgm_v_p = []
        for a in v:
            if numth.jacobi(a, p) == -1:
                val = Val(2, p - 1)
                sgm_v_p.append(val)
            else:
                # print(f'a_base item {a},{p}')
                ord = Ord(p, a)
                val = Val(2, ord)
                sgm_v_p.append(val)
        return sgm_v_p


def Lambda_p(a_base, p):
    ord_base = []
    for a in a_base:
        ord_base.append(Ord(p, a))
    lambda_p = numth.lcm(ord_base)
    return lambda_p


def Lambda_list(a_base, primes):
    lmd_list = []
    for p in primes:
        lmd_list.append(Lambda_p(a_base, p))
    lmd = numth.lcm(lmd_list)
    return lmd


# расчет "мю"
def Mu_p(a_base, p):
    lambda_p = Lambda_p(a_base, p)
    # print("a_base %s\nord_base %s\n lambda_p = %s" % (a_base, ord_base, lambda_p))
    mu = int((p - 1) / lambda_p)
    return mu


# Проверка равенства сигнатур для списка простых чисел
def check_signs(a_base, primes):
    true_list = []
    # print(primes)
    for i in range(len(primes) - 1):
        # print(f"prime i {primes[i]} {primes[i+1]}")
        if Sign(a_base, primes[i]) == Sign(a_base, primes[i + 1]):
            true_list.append(True)
        else:
            true_list.append(False)
    return all(true_list)


def t_more_3(a_base, B, primes):
    if a_base[-1] < primes[0]:
        if np.prod(primes) * primes[-1] < B:
            if check_signs(a_base, primes):
                for a in a_base:
                    return


# проверка псевдопростоты если n=pq, где q=2p-1
def psp_2(a_base, pq):
    p, q = pq[0], pq[1]
    if p % 4 == 1:  # Remark for psp(2,n). Так как ищем для нескольких баз, то учитываем сразу
        for a in a_base[1:]:
            if numth.jacobi(q, a) != 1:
                return False
            else:
                continue
    else:
        return False
    return True


# проверка псевдопростоты по одной базе
def check_for_psp(a, n):
    mod = powmod(a, n - 1, n)
    if mod == 1:
        return True
    else:
        return False


# проверка псевдопростоты числа по нескольким базам
def psp(a_base, n):
    arr = []
    for a in a_base:
        arr.append(check_for_psp(a, n))
    return all(arr)


def t_2(a_base, B_l, B_h, primes_list):
    clearfile(f"res/jae/2/{a_base}/spsp_{B_l}_{B_h}.txt")
    spsp = []
    ### Посчет времени работы
    start_time = time.time()
    ###
    i = 1
    for p in primes_list:
        if p < int(root(B_h, 2)) and p > int(root(B_l, 2)):
            if p > a_base[-1]:
                lmd_p = Lambda_p(a_base, p)
                lmd = numth.lcm(lmd_p, 2)
                for k in range(int(1 + (p - 1) / lmd), int((B_h - p) / (p * lmd)), 1):
                    q = 1 + k * lmd
                    #print(q)
                    if numth.is_prime(q) and q > p:

                        if q + 1 == 2 * p:
                            if psp_2(a_base, [p, q]) == True and check_signs(a_base, [p, q]) == True:
                                item = Signature(Sign(a_base, p), [p, q])
                                s = f"{i}   {np.prod(item.primes)}    {item.primes}   {item.sign}\n"
                                writefile(f"res/jae/2/{a_base}/spsp_{B_l}_{B_h}.txt", s)
                                i+=1
                                spsp.append(item)
                            else:
                                continue

                        else:
                            P = p * (1 + k * lmd)
                            if psp(a_base, P) and check_signs(a_base, [p, q]):
                                item = Signature(Sign(a_base, p), [p, q])
                                s = f"{i}   {np.prod(item.primes)}    {item.primes}   {item.sign}\n"
                                writefile(f"res/jae/2/{a_base}/spsp_{B_l}_{B_h}.txt", s)
                                i+=1
                                spsp.append(item)

    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###
    writefile(f"res/jae/2/{a_base}/spsp_{B_l}_{B_h}.txt", total_time)
    return spsp


if __name__ == "__main__":
    #t_2(bases[:2], 0, 10 ** 6, primes)
    #t_2(bases[:2], 10 ** 6, 10 ** 8, primes)
    t_2(bases[:2], 10 ** 8, 10 ** 10, primes)

    # t_2(bases[:2], 10 ** 10, 10 ** 12, primes)
    # t_2(bases[:2], 10 ** 12, 10 ** 14, primes)
    # t_2(bases[:2], 10 ** 14, 10 ** 16, primes)
