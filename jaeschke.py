import time
from gmpy2 import root, gcd

import ecdsa.numbertheory as numth
import numpy as np
from numba import jit

from utils import powmod, readfile, clearfile, Signature, writefile, combinations, parsefile

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

primes = readfile("primes/primes_1m.txt")


@jit
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


@jit
def Val(p, n):
    # p is prime
    # n is integer
    # e is the greatest power of p in n
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


# вычисление сигнатуры
@jit
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


@jit
def Lambda_p(a_base, p):
    ord_base = []
    for a in a_base:
        ord_base.append(Ord(p, a))
    lambda_p = numth.lcm(ord_base)
    return lambda_p


@jit
def Lambda_list(a_base, primes):
    lmd_list = []
    for p in primes:
        lmd_list.append(Lambda_p(a_base, p))
    lmd = numth.lcm(lmd_list)
    return lmd


# расчет "мю"
@jit
def Mu_p(a_base, p):
    lambda_p = Lambda_p(a_base, p)
    # print("a_base %s\nord_base %s\n lambda_p = %s" % (a_base, ord_base, lambda_p))
    mu = int((p - 1) / lambda_p)
    return mu


# Вычисление одинаковых сигнатур

def find_equal_signs(a_base, primes_list):
    clearfile(f"lib/equal/{a_base}/equal_signs.txt")
    clearfile(f"lib/equal/{a_base}/total_time.txt")

    ### Посчет времени работы
    start_time = time.time()
    ###
    signs_list = []  # так как нельзя вернуть словарь с ключом-списком, заводим список сигнатур
    primes_dict = {}  # ключами являются индексы в списке сигнатур
    for prime in primes_list[len(a_base):]:
        print("finding equal ... %s" % (prime))
        sign = Sign(a_base, prime)
        if sign in signs_list:
            primes_dict[signs_list.index(sign)].append(prime)
        else:
            signs_list.append(sign)
            primes_dict[signs_list.index(sign)] = [prime]
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ### Преобразование по классу
    equal_list = []
    for j in range(len(signs_list)):
        temp = Signature(signs_list[j], primes_dict[j])
        equal_list.append(temp)

    ###Запись в файл
    tot_s = total_time
    writefile(f"lib/equal/{a_base}/total_time.txt", tot_s)

    s = ""
    for j in range(len(signs_list)):
        s += f"{j}    {equal_list[j].sign}     {equal_list[j].primes}\n"
    writefile(f"lib/equal/{a_base}/equal_signs.txt", s)

    return equal_list


# Фильтр по t и B
def screen_by_t(a_base, B, t, equal_list):
    clearfile(f"lib/{B}/{t}/t_signs_{t}_{B}.txt")

    ### Посчет времени работы
    start_time = time.time()
    ###

    screening_list = []
    for item in equal_list:  # item - простые числа с одинаковой сигнатурой
        if len(item.primes) >= t - 1 and item.primes[0] > a_base[-1]:
            # берем больше, так как позже будем проверять по группам p1*p2*...*p(t-1)^2<B
            combine = combinations(item.primes, t - 1)  # в порядке возрастания
            for prms in combine:
                prod = np.prod(prms) * prms[-1]
                if prod < B:
                    screening_list.append(Signature(item.sign, prms))

    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ### Запись в файл
    for j in range(len(screening_list)):
        s = f"{j}    {screening_list[j].sign}     {screening_list[j].primes}\n"
        writefile(f"lib/{B}/{t}/t_signs_{t}_{B}.txt", s)
    writefile(f"lib/{B}/{t}/t_signs_{t}_{B}.txt", total_time)

    return screening_list


# Проверка равенства сигнатур для списка простых чисел
@jit
def check_signs(a_base, primes):
    true_list = []
    for i in range(len(primes) - 1):
        if Sign(a_base, primes[i]) == Sign(a_base, primes[i + 1]):
            true_list.append(True)
        else:
            true_list.append(False)
    return all(true_list)


# проверка псевдопростоты если n=pq, где q=2p-1
@jit
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
@jit
def check_for_psp(a, n):
    mod = powmod(a, n - 1, n)
    if mod == 1:
        return True
    else:
        return False


# проверка псевдопростоты числа по нескольким базам
@jit
def psp(a_base, n):
    arr = []
    for a in a_base:
        arr.append(check_for_psp(a, n))
    return all(arr)


def t_2(a_base, B, primes_list):
    clearfile(f"res/jae/2/{a_base}/spsp_{B}.txt")
    spsp = []
    ### Посчет времени работы
    start_time = time.time()
    ###
    i = 1
    for p in primes_list:
        if p < int(root(B, 2)):
            if p > a_base[-1]:
                lmd_p = Lambda_p(a_base, p)
                lmd = numth.lcm(lmd_p, 2)
                for k in range(int(1 + (p - 1) / lmd), int((B - p) / (p * lmd))+1, 1):
                    q = 1 + k * lmd
                    if p * q <= B:# and p * q > B // 100:
                        if numth.is_prime(q) and q > p:
                            if q + 1 == 2 * p:
                                if check_signs(a_base, [p, q]) and psp_2(a_base, [p, q]):
                                    item = Signature(Sign(a_base, p), [p, q])
                                    s = f"{i}    {np.prod(item.primes)}    {item.primes}    {item.sign}\n"
                                    writefile(f"res/jae/2/{a_base}/spsp_{B}.txt", s)
                                    i += 1
                                    spsp.append(item)
                                else:
                                    continue

                            else:
                                P = p * (1 + k * lmd)
                                if psp(a_base, P) and check_signs(a_base, [p, q]):
                                    item = Signature(Sign(a_base, p), [p, q])
                                    s = f"{i}    {np.prod(item.primes)}    {item.primes}    {item.sign}\n"
                                    writefile(f"res/jae/2/{a_base}/spsp_{B}.txt", s)
                                    i += 1
                                    spsp.append(item)
                    # else:
                    # break
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###
    writefile(f"res/jae/2/{a_base}/spsp_{B}.txt", total_time)
    return spsp


def t_more_3(a_base, B, t, primes_list):
    clearfile(f"res/jae/{t}/{a_base}/spsp_{B}.txt")
    spsp = []
    ### Посчет времени работы
    start_time = time.time()
    ###
    i = 1
    equal_list = parsefile(f"lib/equal/{a_base}/equal_signs.txt")

    for item in equal_list:  # item - простые числа с одинаковой сигнатурой
        if len(item.primes) >= t - 1 and item.primes[0] > a_base[-1]:
            # берем больше, так как позже будем проверять по группам p1*p2*...*p(t-1)^2<B
            combine = combinations(item.primes, t - 1)  # в порядке возрастания
            for prms in combine:
                prod = np.prod(prms)
                if prod * prms[-1] < B:
                    a = a_base[0]
                    mu = Lambda_list([a], prms)
                    if gcd(mu, prod) > 1:
                        continue
                    else:
                        import gmpy2
                        c = gmpy2.powmod(prod, -1, mu)
                        for pt in primes_list:
                            if pt > prms[-1] and pt <= B / prod and pt % mu == c:
                                if psp(a_base, pt * prod) and check_signs(a_base, [pt, prms[-1]]):
                                    item = Signature(Sign(a_base, pt), prms + [pt])
                                    s = f"{i}    {np.prod(item.primes)}    {item.primes}    {item.sign}\n"
                                    writefile(f"res/jae/{t}/{a_base}/spsp_{B}.txt", s)
                                    i += 1
                                    spsp.append(item)

                else:
                    break  # к другому item'у т.к. combine упорядочен вертикально и горизонтально

    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###
    writefile(f"res/jae/{t}/{a_base}/spsp_{B}.txt", total_time)
    return spsp


def run_t_2():
    for i in range(6, 10, 2):
        for j in range(2,5):
            print(i, j)
            t_2(bases[:j], 10 ** i, primes)


def run_t_3(t, primes_list):
    for i in range(6, 12, 2):
        for j in range(2,5):
            print(i, j)
            t_more_3(bases[:j], 10 ** i, t, primes_list)


def equal_signs():
    print(bases[:2])
    find_equal_signs(bases[:2], primes)
    print(bases[:3])
    find_equal_signs(bases[:3], primes)
    print(bases[:4])
    find_equal_signs(bases[:4], primes)
    print(bases[:5])
    find_equal_signs(bases[:5], primes)


if __name__ == "__main__":
    print()
