import sys
import time

import ecdsa.numbertheory as numth
import numpy as np
from gmpy2 import root, gcd

# from memory_profiler import profile
from jaeschke import Mu_p, Sign, Lambda_list, Lambda_p, Val, psp, screen_by_t
from utils import readfile, writefile, clearfile, powmod, Signature, parsefile

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

primes = readfile("primes/primes_1m.txt")


# Нечетные числа, свободные от квадратов
def step0(Q11):
    clearfile(f"lib/free_sqr.txt")
    start_time = time.time()
    free_sqr = []
    for n in range(3, Q11, 2):
        print(n)
        arr = numth.factorization(n)
        power = 1
        for i in range(len(arr)):
            power *= arr[i][1]
        if power == 1:
            free_sqr.append(n)
            writefile(f"lib/free_sqr.txt", f"{n} ")
    total_time = "\n--- %s seconds ---\n" % (time.time() - start_time)
    writefile(f"lib/free_sqr.txt", total_time)

    return free_sqr


# Вычисляем мю для p
def step1(a_base, B, primes):
    clearfile(f"lib/mu/{a_base}/total_time.txt")
    ### Посчет времени работы
    start_time = time.time()
    ###
    primes_dict = {}
    for prime in primes[len(a_base):]:
        if prime < B:  # ограничение вычислений
            mu = Mu_p(a_base, prime)
            print(prime, mu)
            # if mu == 1:
            #    continue
            if mu in primes_dict.keys():
                primes_dict[mu].append(prime)
            else:
                primes_dict[mu] = [prime]
        else:
            break
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    tot_s = total_time
    writefile(f"lib/mu/{a_base}/total_time.txt", tot_s)

    for item in primes_dict:
        clearfile(f"lib/mu/{a_base}/mu_{item}.txt")
        s = ''.join(str(l) + ' ' + '\n' * (n % 8 == 7) for n, l in enumerate(primes_dict[item]))
        writefile(f"lib/mu/{a_base}/mu_{item}.txt", s)

    return primes_dict


# Проверка равенства сигнатур для списка простых чисел
def check_signs(a_base, primes):
    true_list = []
    # print(primes)
    for i in range(len(primes) - 1):
        # print(f"prime i {primes[i]}    {primes[i+1]}")
        if Sign(a_base, primes[i]) == Sign(a_base, primes[i + 1]):
            true_list.append(True)
        else:
            true_list.append(False)
    return all(true_list)


# Удовлетворение условиям p_next
def checking_p(b, lmd, B_h):
    p3_next_list = []
    ###Случай подбора
    r = powmod(b, -1, lmd)
    p = r

    if primes[-1] < B_h / b:
        bound = primes[-1]
    else:
        bound = B_h / b

    while p < bound:
        if numth.is_prime(int(p)):
            p3_next_list.append(p)
        p += lmd
    return p3_next_list


# Нахождение последующего p для t>2
def next_p(p_exist, a_base, B):
    b = int(np.prod(p_exist))
    if check_signs(a_base, p_exist) and b / p_exist[-1] < B / p_exist[-2]:
        lmd = Lambda_list(a_base, p_exist)
        print(f"lmd {lmd}")
        if gcd(b, lmd) == 1:
            p_next_list = checking_p(b, lmd, B)
            # print(f"p_next_list {p_next_list}")
            return sorted(p_next_list)
        elif gcd(b, lmd) != 1:
            return "Impossible n sequences"


# @profile
# @memprof(plot=True)
def step_t_2(a_base, B, primes_list):
    clearfile(f"res/jnd/2/{a_base}/spsp_{B//100}_{B}.txt")
    clearfile(f"res/jnd/2/{a_base}/n_list_{B//100}_{B}.txt")
    n_list = []
    ### Посчет времени работы
    start_time = time.time()
    ###
    for p1 in primes_list:
        p1 = int(p1)

        if p1 < int(root(B, 2)):
            if p1 > a_base[-1]:

                s = ""
                if p1 < 10 ** 6:
                    gcd_23 = int(gcd(2 ** (p1 - 1) - 1, 3 ** (p1 - 1) - 1))
                    factors = sorted(numth.factorization(gcd_23))
                    for i in range(len(factors)):
                        p2 = factors[i][0]
                        if p2 * p1 <= B and p1 * p2 > B // 100:
                            if p2 > p1:  # В дальнейшем для того, чтобы числа в интервалах не повторялись
                                signss = check_signs(a_base, [p1, p2])
                                if signss:
                                    item = Signature(Sign(a_base, p1), [p1, p2])
                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                    n_list.append(item)
                        # else:
                        # break
                elif p1 > 10 ** 8:
                    lmd_p = Lambda_p(a_base, p1)  # lmd_p = p1-1
                    p2 = 1

                    while p2 <= p1 and p1 * p2 < B // 100:  # к условию, что p2>p1
                        p2 += lmd_p

                    while p2 * p1 <= B and p1 * p2 > B // 100:
                        signss = check_signs(a_base, [p1, p2])
                        if signss:
                            item = Signature(Sign(a_base, p1), [p1, p2])
                            s += f"{item.primes}    {signss}    {item.sign}\n"
                            n_list.append(Signature(item.sign, [p1, p2]))
                        p2 += lmd_p
                else:  # между 10**6..10**8
                    if len(a_base) > 6:
                        a_base = a_base[:6]
                    lmd_p = Lambda_p(a_base, p1)
                    leg1 = []
                    for a in a_base:
                        leg1.append(numth.jacobi(a, p1))

                    if p1 % 4 == 1:
                        p2_4k3 = readfile("primes/4k+3.txt")
                        p2_4k1 = readfile("primes/4k+1.txt")

                        for p2 in p2_4k3:
                            if p1 * p2 <= B and p1 * p2 > B // 100:
                                if p2 % lmd_p == 1 and p2 > p1:
                                    leg2 = []
                                    for a in a_base:
                                        leg2.append(numth.jacobi(a, p2))
                                    signss = check_signs(a_base, [p1, p2])
                                    if leg1 == leg2 and signss:
                                        item = Signature(Sign(a_base, p1), [p1, p2])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                            # else:
                            # break
                        for p2 in p2_4k1:
                            if p1 * p2 <= B and p1 * p2 > B // 100:
                                if p2 % lmd_p == 1 and p2 > p1:
                                    leg2 = []
                                    for a in a_base:
                                        leg2.append(numth.jacobi(a, p2))
                                    signss = check_signs(a_base, [p1, p2])
                                    if np.prod(leg2) == 1 and signss:  # если все 1, то произведение 1
                                        item = Signature(Sign(a_base, p1), [p1, p2])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                            # else:
                            # break
                    elif p1 % 8 == 5:
                        p2_8k5 = readfile("primes/8k+5.txt")
                        p2_8k1 = readfile("primes/8k+1.txt")

                        for p2 in p2_8k5:
                            if p1 * p2 <= B and p1 * p2 > B // 100:
                                if p2 % lmd_p == 5 and p2 > p1:
                                    leg2 = []
                                    for a in a_base:
                                        leg2.append(numth.jacobi(a, p2))
                                    signss = check_signs(a_base, [p1, p2])
                                    if leg1 == leg2 and signss:
                                        item = Signature(Sign(a_base, p1), [p1, p2])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                            # else:
                            # break
                        for p2 in p2_8k1:
                            if p1 * p2 <= B and p1 * p2 > B // 100:
                                if p2 % lmd_p == 1 and p2 > p1:
                                    leg2 = []
                                    for a in a_base:
                                        leg2.append(numth.jacobi(a, p2))
                                    signss = check_signs(a_base, [p1, p2])
                                    if np.prod(leg2) == 1 and signss:
                                        item = Signature(Sign(a_base, p1), [p1, p2])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                            # else:
                            # break
                    elif p1 % 8 == 1:
                        sign = Sign([2], p1)[0]
                        e, f = Val(2, p1 - 1), Val(2, sign)

                        for p2 in primes:
                            if p1 * p2 <= B and p1 * p2 > B // 100:
                                if p2 > p1 and e == f:
                                    if p2 % (2 ** (e - 1)) == 2 ** e % (2 ** (e - 1)) and p2 % lmd_p == 1:
                                        leg2 = []
                                        for a in a_base:
                                            leg2.append(numth.jacobi(a, p2))
                                        signss = check_signs(a_base, [p1, p2])
                                        if leg1 == leg2 and signss:
                                            item = Signature(Sign(a_base, p1), [p1, p2])
                                            s += f"{item.primes}    {signss}    {item.sign}\n"
                                            n_list.append(item)
                                    elif p2 % 2 ** (e + 1) == 1 and p2 % lmd_p == 1:
                                        leg2 = []
                                        for a in a_base:
                                            leg2.append(numth.jacobi(a, p2))
                                        signss = check_signs(a_base, [p1, p2])
                                        if np.prod(leg2) == 1 and signss:
                                            item = Signature(Sign(a_base, p1), [p1, p2])
                                            s += f"{item.primes}    {signss}    {item.sign}\n"
                                            n_list.append(item)
                                elif p2 > p1 and f < e and p1 * p2 <= B and p1 * p2 > B // 100:
                                    signss = check_signs(a_base, [p1, p2])
                                    if p2 % lmd_p == 1 and signss:
                                        item = Signature(Sign(a_base, p1), [p1, p2])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                            # else:
                            # break
                writefile(f"res/jnd/2/{a_base}/n_list_{B//100}_{B}.txt", s)
        else:
            continue

    i = 1
    spsp = []
    ss = ""
    for item in n_list:
        prod = np.prod(item.primes)
        if psp(a_base, prod):
            ss += f"{i}    {prod}    {item.primes}    {item.sign}\n"
            i += 1
            spsp.append(item)
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ss += f"{total_time}\n"
    writefile(f"res/jnd/2/{a_base}/spsp_{B//100}_{B}.txt", ss)
    return np.array(spsp)


def step_t_3(a_base, B):
    clearfile(f"res/jnd/3/{a_base}/spsp_{B//100}_{B}.txt")
    clearfile(f"res/jnd/3/{a_base}/n_list_{B//100}_{B}.txt")
    n_list = []

    ### Посчет времени работы
    start_time = time.time()
    ###

    equal_2_list = parsefile(f"res/jnd/2/{a_base}/n_list_{B//100}_{B}.txt")
    # упорядочены по возрастанию p1, где p1<p2
    if len(equal_2_list) != 0:
        for i in range(len(equal_2_list)):
            p1 = equal_2_list[i].primes[0]
            p2 = equal_2_list[i].primes[1]
            b = int(p1 * p2)
            if p1 <= int(root(B, 3)) and b * p2 < B:

                s = ""
                if len(a_base) > 6:
                    a_base = a_base[:6]
                leg1 = []

                for a in a_base:
                    leg1.append(numth.jacobi(a, p1))

                p2_3k4 = readfile("primes/4k+3.txt")
                p2_1k4 = readfile("primes/4k+1.txt")
                p2_5k8 = readfile("primes/8k+5.txt")
                p2_1k8 = readfile("primes/8k+1.txt")

                if p1 % 4 == 3:
                    print(f"p1 34 {p1}")
                    if p2 in p2_3k4 and p2 > p1:  # на всякий случай проверим
                        print(f"p2 34 {p2}")
                        leg2 = []
                        for a in a_base:
                            leg2.append(numth.jacobi(a, p2))
                        if leg1 == leg2:  # Prop.2 inverse is true
                            if b < 2 * 10 ** 6:  # a trick
                                gcd_23 = int(gcd(2 ** (b - 1) - 1, 3 ** (b - 1) - 1))
                                factor_list = numth.factorization(gcd_23)
                                for i in range(len(factor_list)):
                                    p3 = factor_list[i][0]
                                    if p3 * b <= B and p3 * b > B // 100:
                                        if p3 > p2:
                                            f"p3 {p3}"
                                            signss = check_signs(a_base, [p1, p3])
                                            if signss:
                                                item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                s += f"{item.primes}    {signss}    {item.sign}\n"
                                                n_list.append(item)
                            else:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base, B)  # ищем подходящие p3
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 * b <= B and p3 * b > B // 100:
                                            if p3 > p2:
                                                f"p3 {p3}"
                                                signss = check_signs(a_base, [p1, p3])
                                                if signss:
                                                    item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                                    n_list.append(item)
                                else:
                                    continue  # новый item из equal_2_list
                        else:
                            continue

                    elif p2 in p2_1k4:
                        print(f"p2 14 to mu {p2}")
                        if Mu_p(a_base, p2) == 4:
                            pass  # переход к mu=4

                elif p1 % 8 == 5:
                    print(f"p1 58 {p1}")

                    if p2 in p2_5k8 and p2 > p1:
                        print(f"p2 58 {p2}")
                        if len(a_base) > 5:
                            a_base = a_base[:5]

                        leg2 = []
                        for a in a_base:
                            leg2.append(numth.jacobi(a, p2))
                        if leg1 == leg2:
                            p_exist = np.array([p1, p2])
                            p3_list = next_p(p_exist, a_base, B)
                            if isinstance(p3_list, list):
                                for p3 in p3_list:
                                    if p3 * b <= B and p3 * b > B // 100:
                                        if p3 > p2:
                                            f"p3 {p3}"
                                            signss = check_signs(a_base, [p1, p3])
                                            if signss:
                                                item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                s += f"{item.primes}    {signss}    {item.sign}\n"
                                                n_list.append(item)
                            else:
                                continue
                        else:
                            continue
                    if p2 in p2_1k8 and p2 > p1:
                        print(f"p2 18 {p2}")
                        if p2 % 16 == 9:
                            print(f"p2 916 {p2}")
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            if np.prod(leg2) == 1 and p2 > p1:  # если все 1, то произведение 1
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base, B)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 * b <= B and p3 * b > B // 100:
                                            if p3 > p2:
                                                f"p3 {p3}"
                                                signss = check_signs(a_base, [p1, p3])
                                                if signss:
                                                    item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                                    n_list.append(item)
                                else:
                                    continue
                            else:
                                continue

                        elif p2 % 16 == 1:
                            print(f"p2 116 to mu {p2}")
                            if Mu_p(a_base, p2) == 4:
                                pass  # переход к mu=4

                    if p2 in p2_3k4 and p2 > p1:  # в тексте этого нет, но на всякий случай проверим
                        print(f"p2 34 {p2}")
                        p_exist = np.array([p1, p2])
                        p3_list = next_p(p_exist, a_base, B)
                        # print(f"p3 list {p3_list}")
                        if isinstance(p3_list, list):
                            for p3 in p3_list:
                                if p3 * b <= B and p3 * b > B // 100:
                                    if p3 > p2:
                                        print(f"p3 {p3}")
                                        signss = check_signs(a_base, [p1, p3])
                                        if signss:
                                            item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                            s += f"{item.primes}    {signss}    {item.sign}\n"
                                            n_list.append(item)
                        else:
                            continue

                elif p1 % 8 == 1:
                    print(f"p1 18 {p1} p2 any {p2}")
                    e, f = Val(2, p1 - 1), Val(2, Lambda_p(a_base, p1))

                    if len(a_base) > 5:
                        a_base = a_base[:5]

                    if p2 > p1 and e == f:
                        if p2 % (2 ** (e + 1)) == (1 + 2 ** e) % (2 ** (e + 1)):  # !!!! СКОБКИ???
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            if leg1 == leg2:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base, B)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 * b <= B and p3 * b > B // 100:
                                            if p3 > p2:
                                                f"p3 {p3}"
                                                signss = check_signs(a_base, [p1, p3])
                                                if signss:
                                                    item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                                    n_list.append(item)
                                else:
                                    continue
                            else:
                                continue

                        elif p2 % (2 ** (e + 2)) == 1:
                            if Mu_p(a_base, p2) == 4:
                                pass  # переход к mu=4

                        elif p2 % (2 ** (e + 2)) != 1 and p2 % (2 ** (e + 2)) == (1 + 2 ** (e + 1)) % 2 ** (
                                e + 2):  # !!!! СКОБКИ???
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            if np.prod(leg2) == 1:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base, B)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 * b <= B and p3 * b > B // 100:
                                            if p3 > p2:
                                                f"p3 {p3}"
                                                signss = check_signs(a_base, [p1, p3])
                                                if signss:
                                                    item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                                    n_list.append(item)
                                else:
                                    continue
                            else:
                                continue

                    elif p2 > p1 and f < e:
                        if p2 % 2 ** f == p1:
                            if f == e - 1 and Mu_p(a_base, p1) == 2:  # это есть условие выше
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base, B)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 * b <= B and p3 * b > B // 100:
                                            if p3 > p2:
                                                f"p3 {p3}"
                                                signss = check_signs(a_base, [p1, p3])
                                                if signss:
                                                    item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                                    n_list.append(item)
                                else:
                                    continue
                            else:
                                continue
                        else:
                            continue

                # p1 is any in primes
                mu_4 = readfile(f"lib/mu/{a_base}/mu_4.txt")
                if p2 in mu_4 and p2 > p1:  # если p2 mu=4, то не обязательно чтобы и p1 mu=4
                    print(f"p2 mu4 {p2}")
                    p_exist = np.array([p1, p2])
                    p3_list = next_p(p_exist, a_base, B)
                    if isinstance(p3_list, list):
                        for p3 in p3_list:
                            if p3 * b <= B and p3 * b > B // 100:
                                if p3 > p2:
                                    f"p3 {p3}"
                                    signss = check_signs(a_base, [p1, p3])
                                    if signss:
                                        item = Signature(Sign(a_base, p1), [p1, p2, p3])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                    else:
                        continue
                else:
                    continue

                writefile(f"res/jnd/3/{a_base}/n_list_{B//100}_{B}.txt", s)
            else:
                continue

    i = 1
    spsp = []
    ss = ""
    for item in n_list:
        prod = np.prod(item.primes)
        if psp(a_base, prod):
            ss += f"{i}    {prod}    {item.primes}    {item.sign}\n"
            i += 1
            spsp.append(item)
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ss += f"{total_time}\n"
    writefile(f"res/jnd/3/{a_base}/spsp_{B//100}_{B}.txt", ss)
    return np.array(spsp)


def step_t_4(a_base, B):
    clearfile(f"res/jnd/4/{a_base}/spsp_{B//100}_{B}.txt")
    clearfile(f"res/jnd/4/{a_base}/n_list_{B//100}_{B}.txt")
    n_list = []

    ### Посчет времени работы
    start_time = time.time()
    ###

    equal_3_list = parsefile(f"res/jnd/3/{a_base}/n_list_{B//100}_{B}.txt")
    # упорядочены по возрастанию p1, где p1<p2
    if len(equal_3_list) != 0:
        for i in range(len(equal_3_list)):
            p1 = equal_3_list[i].primes[0]
            p2 = equal_3_list[i].primes[1]
            p3 = equal_3_list[i].primes[2]
            b = p1 * p2 * p3
            if p1 <= int(root(B, 4)) and b * p3 < B:
                s = ""
                if p1 % 4 == 3:  # Вместо сигнатур вычисляется символ Лежандра
                    p4_3k4 = readfile("primes/4k+3.txt")
                    for p4 in p4_3k4:
                        if p4 * b <= B and p4 * b > B // 100:
                            if p4 > p3:
                                f"p4 {p4}"
                                signss = check_signs(a_base, [p1, p4])
                                if signss:
                                    item = Signature(Sign(a_base, p1), [p1, p2, p3, p4])
                                    s += f"{item.primes}    {signss}    {item.sign}\n"
                                    n_list.append(item)

                else:
                    p_exist = np.array([p1, p2, p3])
                    p4_list = next_p(p_exist, a_base, B)
                    if isinstance(p4_list, list):
                        for p4 in p4_list:
                            if p4 * b <= B and p4 * b > B // 100:
                                if p4 > p3:
                                    f"p4 {p4}"
                                    signss = check_signs(a_base, [p1, p4])
                                    if signss:
                                        item = Signature(Sign(a_base, p1), [p1, p2, p3, p4])
                                        s += f"{item.primes}    {signss}    {item.sign}\n"
                                        n_list.append(item)
                writefile(f"res/jnd/4/{a_base}/n_list_{B//100}_{B}.txt", s)
            else:
                continue
    i = 1
    spsp = []
    ss = ""
    for item in n_list:
        prod = np.prod(item.primes)
        if psp(a_base, prod):
            ss += f"{i}    {prod}    {item.primes}    {item.sign}\n"
            i += 1
            spsp.append(item)
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ss += f"{total_time}\n"
    writefile(f"res/jnd/4/{a_base}/spsp_{B//100}_{B}.txt", ss)
    return np.array(spsp)


def step_t_5(a_base, B, p1):
    if p1 <= int(root(B, 5)):
        ### Посчет времени работы
        start_time = time.time()
        ###
        n_list = []
        l = 5

        if p1 % 4 == 3:
            n_list = [p1]
            q_3k4 = readfile("primes/4k+3.txt")
            sign_p1 = []
            for a in a_base:
                sign_p1.append(numth.jacobi(a, p1))

            for q in q_3k4:
                if q % 24 == p1 % 24:
                    sign_q = []
                    for a in a_base:
                        sign_q.append(numth.jacobi(a, q))
                    if sign_q == sign_p1:
                        n_list.append(q)
        # else:
        # for p in primes:

        ###
        total_time = "--- %s seconds ---\n" % (time.time() - start_time)
        ###
        return np.array(n_list)
    else:
        print(f"Value Error: p1 > {int(root(B, 5))}")


def step2(t, a_base, B, equal_list):
    clearfile(f"{B}/{t}/l_signs_{t}_{B}.txt")

    B_root = int(root(B, t))
    if B > B_root:
        B = B_root

    if t >= 5:  # вместо умножения и последующео сравнения будем делить, ибо Memory Error
        l = 5
        screening_list = screen_by_t(t, a_base, B, equal_list)
        ### Посчет времени работы
        start_time = time.time()
        ###
        n_list = []
        for item in screening_list:
            scr_primes = item.primes

            if scr_primes[0] % 4 == 3 and scr_primes[0] <= B:
                # вычисление сигнатур через символ лежандра
                base_prod = np.prod(a_base)
                sign_p = numth.jacobi(base_prod, scr_primes[0])

                k = 0
                while scr_primes[0] + 24 * k < B:
                    n_list.append(scr_primes[0] + 24 * k)
                    k += 1

            else:
                P1 = B
                for i in range(5):
                    P1 /= scr_primes[i]
                while l <= len(scr_primes):
                    P2 = P1 * scr_primes[4] / scr_primes[l]
                    if 1 <= B and 1 > P2:
                        temp = Signature(item.sign, scr_primes)
                        n_list.append(temp)
                        break
                    l += 1
        ###
        total_time = ("--- %s seconds ---" % (time.time() - start_time))
        ###

        ### Запись в файл
        s = total_time
        for j in range(len(n_list)):
            s += f"{j}    {n_list[j].sign}    {n_list[j].primes}\n"
        writefile(f"{B}/{t}/l_signs_{t}_{B}.txt", s)

        return n_list  ###6 последовательностей, 0 spsp


def run_t_2(base_len):
    print()
    # Готовы
    for i in range(6, 12, 2):
        print(i)
        step_t_2(bases[:base_len], 10 ** i, primes)

    # Не готовы
    # for i in range(10, 18, 2):
    #    step_t_2(bases[:base_len], 10 ** i, primes)


def run_t_3(base_len):
    print()
    # Готовы
    for i in range(6, 12, 2):
        step_t_3(bases[:base_len], 10 ** i)

    # Не готовы
    # for i in range(10, 18, 2):
    #    step_t_3(bases[:base_len], 10 ** i, primes)


def run_t_4(base_len):
    print()

    # Готовы
    #for i in range(6, 12, 2):
    #    step_t_2(bases[:base_len], 10 ** i)

    # Не готовы
    #for i in range(10, 18, 2):
    #    step_t_2(bases[:base_len], 10 ** i, primes)


if __name__ == "__main__":
    print(sys.maxsize)
    print(sys.version)
    # print((list(primes)).index(9999677)) #664560
    # print((list(primes)).index(999491)) #78464

    #run_t_2(4)
    run_t_3(4)
    # run_t_4(2)
    # run_t_5(2)
