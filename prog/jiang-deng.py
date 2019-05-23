import time
from gmpy2 import root, gcd

import ecdsa.numbertheory as numth
import numpy as np

from prog.utils import readfile, writefile, clearfile, powmod

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

Q11 = 3825123056546413051
primes = readfile("primes/primes_1m.txt")


class Signature():
    def __init__(self, sign, primes):
        self.sign = sign
        self.primes = primes


def Primes_modulo(primes):
    """
    clearfile("primes/4k+1.txt")
    clearfile("primes/4k+3.txt")
    clearfile("primes/8k+1.txt")
    clearfile("primes/8k+5.txt")
    clearfile("primes/else.txt")
    """

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

    """
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
    """
    return len(np.array(p_1mod4)), len(np.array(p_3mod4)), len(np.array(p_1mod8)), len(np.array(p_5mod8)), len(
        np.array(p_else))


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


# проверка псевдопростоты по одной базе
def check_for_psp(a, p):
    mod = powmod(a, p - 1, p)
    # print(mod)
    if mod == 1:
        return True
    else:
        return False


# проверка псевдопростоты числа по нескольким базам
def psp(v, p):
    arr = []
    for a in v:
        arr.append(check_for_psp(a, p))
    return all(arr)


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
def step1(a_base):
    clearfile(f"lib/mu_{a_base}.txt")
    B = int(root(Q11, 2))

    ### Посчет времени работы
    start_time = time.time()
    ###
    primes_dict = {}
    for prime in primes[len(a_base):]:
        if prime < B:  # ограничение вычислений
            mu = Mu_p(a_base, prime)
            print(prime, mu)
            if mu == 1:
                continue
            if mu in primes_dict.keys():
                primes_dict[mu].append(prime)
            else:
                primes_dict[mu] = [prime]
        else:
            break
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###
    s = total_time
    for item in primes_dict:
        s += f"{mu}    {item[mu]}\n"
    writefile(f"lib/mu_{a_base}.txt", s)

    return primes_dict


# Вычисление одинаковых сигнатур
def find_equal_signs(a_base, primes):
    clearfile(f"lib/equal_signs_{a_base}.txt")

    ### Посчет времени работы
    start_time = time.time()
    ###
    signs_list = []  # так как нельзя вернуть словарь с ключом-списком, заводим список сигнатур
    primes_dict = {}  # ключами являются индексы в списке сигнатур
    for prime in primes[len(a_base):]:
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
    s = total_time
    for j in range(len(signs_list)):
        s += f"{j}   {equal_list[j].sign}    {equal_list[j].primes}\n"
    writefile(f"lib/equal_signs_{a_base}.txt", s)

    return equal_list


# Фильтр по t
def screen_by_t(t, a_base, B, equal_list):
    clearfile(f"lib/{B}/{t}/t_signs_{t}_{B}.txt")

    ### Посчет времени работы
    start_time = time.time()
    ###
    screening_list = []
    for item in equal_list:  # item - простые числа с одинаковой сигнатурой
        if len(item.primes) >= t:
            temp = Signature(item.sign, item.primes)
            screening_list.append(temp)
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ### Запись в файл
    s = total_time
    for j in range(len(screening_list)):
        s += f"{j}   {screening_list[j].sign}    {screening_list[j].primes}\n"
    writefile(f"lib/{B}/{t}/t_signs_{t}_{B}.txt", s)

    return screening_list


def step2(t, a_base, B, equal_list):
    clearfile(f"{B}/{t}/l_signs_{t}_{B}.txt")

    B_root = int(root(Q11, t))
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
                while scr_primes[0] + 24 * k < Q11:
                    n_list.append(scr_primes[0] + 24 * k)
                    k += 1

            else:
                P1 = Q11
                for i in range(5):
                    P1 /= scr_primes[i]
                while l <= len(scr_primes):
                    P2 = P1 * scr_primes[4] / scr_primes[l]
                    if 1 <= Q11 and 1 > P2:
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
            s += f"{j}   {n_list[j].sign}    {n_list[j].primes}\n"
        writefile(f"{B}/{t}/l_signs_{t}_{B}.txt", s)

        return n_list  ###6 последовательностей, 0 spsp


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


# Удовлетворение условиям p_next
def checking_p(prev_p, b, lmd):
    n = []
    """ ###Случай перебора
    for p in primes_list:
        if p > prev_p and p <= Q11 / b:
            if powmod(b, -1, lmd) == p:
                n.append(b * p)
        else:
            break
    """
    ###Случай подбора
    r = powmod(b, -1, lmd)
    i = int(prev_p / r)
    if i < 1:
        i = 1
    p = i * lmd + r
    while p < Q11 / b:
        n.append(b * p)
    return n


# Нахождение последующего p для t>2
def next_p(p_exist, a_base):
    b = int(np.prod(p_exist))
    if check_signs(a_base, p_exist) and b / p_exist[-1] < Q11 / p_exist[-2]:
        lmd = Lambda_list(a_base, p_exist)
        # print(lmd)
        if gcd(b, lmd) == 1:
            p_next_list = checking_p(p_exist[-1], b, lmd)
            return p_next_list
        elif gcd(b, lmd) != 1:
            return "Impossible n sequences"


def step_t_2(a_base, primes_list):
    clearfile(f"res/j'n'd/2/spsp_{a_base}_2_{primes_list[0]}.txt")
    clearfile(f"res/j'n'd/2/n_list_{a_base}_2_{primes_list[0]}.txt")
    n_list = []

    ### Посчет времени работы
    start_time = time.time()
    ###
    for p1 in primes_list:
        p1 = int(p1)
        if p1 <= int(root(Q11, 2)):
            s = ""
            if p1 < 10 ** 6:
                gcd_23 = int(gcd(2 ** (p1 - 1) - 1, 3 ** (p1 - 1) - 1))
                factors = numth.factorization(gcd_23)
                for i in range(len(factors)):
                    p2 = factors[i][0]
                    if p2 > p1 and p2 <= Q11 / p1:
                        signss = check_signs(a_base, [p1, p2])
                        if signss:
                            s += f"p1 < 10**6: {p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                            n_list.append([p1, p2])
            elif p1 > 10 ** 8:
                lmd_p = Lambda_p(a_base, p1)  # lmd_p = p1-1
                i = int((p1 - 1) / lmd_p) + 1
                p2 = lmd_p * i + 1
                while p2 <= Q11 / p1:
                    signss = check_signs(a_base, [p1, p2])
                    if signss:
                        s += f"p1 > 10**8: {p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                        n_list.append([p1, p2])
                    p2 += lmd_p
            else:
                if len(a_base) > 6:
                    a_base = a_base[:6]
                s += f"\n10**6 < p1 < 10**8\nchanged to {a_base}\n"
                lmd_p = Lambda_p(a_base, p1)
                leg1 = []
                for a in a_base:
                    leg1.append(numth.jacobi(a, p1))

                if p1 % 4 == 1:
                    p2_4k3 = readfile("primes/4k+3.txt")
                    p2_4k1 = readfile("primes/4k+1.txt")

                    for p2 in p2_4k3:
                        if p2 % lmd_p == 1:
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            signss = check_signs(a_base, [p1, p2])
                            if leg1 == leg2 and signss:
                                s += f"\np1 is 4k+1\np2 is 4k+3\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                n_list.append([p1, p2])

                    for p2 in p2_4k1:
                        if p2 % lmd_p == 1:
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            signss = check_signs(a_base, [p1, p2])
                            if np.prod(leg2) == 1 and signss:  # если все 1, то произведение 1
                                s += f"\np1 is 4k+1\np2 is 4k+1\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                n_list.append([p1, p2])
                elif p1 % 8 == 5:
                    p2_8k5 = readfile("primes/8k+5.txt")
                    p2_8k1 = readfile("primes/8k+1.txt")

                    for p2 in p2_8k5:
                        if p2 % lmd_p == 5:
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            signss = check_signs(a_base, [p1, p2])
                            if leg1 == leg2 and signss:
                                s += f"\np1 is 8k+5\np2 is 8k+5\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                n_list.append([p1, p2])
                    for p2 in p2_8k1:
                        if p2 % lmd_p == 1:
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            signss = check_signs(a_base, [p1, p2])
                            if np.prod(leg2) == 1 and signss:
                                s += f"\np1 is 8k+1\np2 is 8k+5\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                n_list.append([p1, p2])
                elif p1 % 8 == 1:
                    sign = Sign([2], p1)[0]
                    e, f = Val(2, p1 - 1), Val(2, sign)

                    for p2 in primes:
                        if e == f:
                            if p2 % (2 ** (e - 1)) == 2 ** e % (2 ** (e - 1)) and p2 % lmd_p == 1:
                                leg2 = []
                                for a in a_base:
                                    leg2.append(numth.jacobi(a, p2))
                                signss = check_signs(a_base, [p1, p2])
                                if leg1 == leg2 and signss:
                                    s += f"\np1 is 8k+1\ne==f\n leg1==leg2\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                    n_list.append([p1, p2])
                            elif p2 % 2 ** (e + 1) == 1 and p2 % lmd_p == 1:
                                leg2 = []
                                for a in a_base:
                                    leg2.append(numth.jacobi(a, p2))
                                signss = check_signs(a_base, [p1, p2])
                                if np.prod(leg2) == 1 and signss:
                                    s += f"\np1 is 8k+1\ne==f\nleg2==1\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                    n_list.append([p1, p2])
                        elif f < e:
                            signss = check_signs(a_base, [p1, p2])
                            if p2 % lmd_p == 1 and signss:
                                s += f"\np1 is 8k+1\nf<e\n{p1}   {p2}    {signss} {Sign(a_base,p1)}\n"
                                n_list.append([p1, p2])

            writefile(f"res/j'n'd/2/n_list_{a_base}_2_{primes_list[0]}.txt", s)
        else:
            print(f"Value Error: p1 > {int(root(Q11, 2))}\n")
            break

    i = 0
    spsp = []
    ss = ""
    for turple in n_list:
        prod = np.prod(turple)
        if psp(a_base, prod):
            ss += f"{i}    {prod}  {turple}    {Sign(a_base,turple[0])}\n"
            i += 1
            spsp.append(turple)
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ss += f"{total_time}\n"
    writefile(f"res/j'n'd/2/spsp_{a_base}_2_{primes_list[0]}.txt", ss)
    return np.array(spsp)


def step_t_3(a_base, primes_list):
    clearfile(f"res/j'n'd/3/spsp_{a_base}_3_{primes_list[0]}.txt")
    clearfile(f"res/j'n'd/3/n_list_{a_base}_3_{primes_list[0]}.txt")
    n_list = []

    ### Посчет времени работы
    start_time = time.time()
    ###

    for p1 in primes_list:
        p1 = int(p1)
        if p1 <= int(root(Q11, 3)):
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
            # mu_4 = readfile(f"lib/mu4.txt")

            if p1 % 4 == 3:
                for p2 in p2_3k4:
                    if p2 > p1:
                        leg2 = []
                        for a in a_base:
                            leg2.append(numth.jacobi(a, p2))
                        if leg1 == leg2:  # Prop.2 inverse is true
                            b = int(p1 * p2)
                            if b < 2 * 10 ** 6:  # a trick
                                gcd_23 = int(gcd(2 ** (b - 1) - 1, 3 ** (b - 1) - 1))
                                factor_list = numth.factorization(gcd_23)
                                for i in range(len(factor_list)):
                                    p3 = factor_list[i][0]
                                    signss = check_signs(a_base, [p1, p2, p3])
                                    if p3 > p2 and p3 < Q11 / b and signss:
                                        s += f"\np1 is 4k+3\np2 is 4k+3\np1*p2 < 2*10**6 (a trick)\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                        n_list.append([p1, p2, p3])
                            else:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 > p2:
                                            signss = check_signs(a_base, [p1, p2, p3])
                                            if signss:
                                                s += f"\np1 is 4k+3\np2 is 4k+3\np1*p2 > 2*10**6 (not a trick)\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                                n_list.append([p1, p2, p3])
                                else:
                                    continue

                for p2 in p2_1k4:
                    if Mu_p(a_base, p2) == 4:
                        break  # переход к mu=4
            elif p1 % 8 == 5:
                for p2 in p2_1k4:
                    if p2 > p1:
                        leg1 = numth.jacobi(2, p1)
                        if leg1 == -1 and Val(2, Ord(p1, 2)) == 2:
                            p_exist = np.array([p1, p2])
                            p3_list = next_p(p_exist, a_base)
                            if isinstance(p3_list, list):
                                for p3 in p3_list:
                                    if p3 > p2:
                                        signss = check_signs(a_base, [p1, p2, p3])
                                        if signss:
                                            s += f"\np1 is 8k+5\np2 is 4k+1\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                            n_list.append([p1, p2, p3])
                            else:
                                continue

                if len(a_base) > 5:
                    a_base = a_base[:5]
                for p2 in p2_5k8:
                    if p2 > p1:
                        leg2 = []
                        for a in a_base:
                            leg2.append(numth.jacobi(a, p2))
                        if leg1 == leg2:
                            p_exist = np.array([p1, p2])
                            p3_list = next_p(p_exist, a_base)
                            if isinstance(p3_list, list):
                                for p3 in p3_list:
                                    if p3 > p2:
                                        signss = check_signs(a_base, [p1, p2, p3])
                                        if signss and p3 > p2:
                                            s += f"\np1 is 8k+5\np2 is 8k+5\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                            n_list.append([p1, p2, p3])
                            else:
                                continue

                for p2 in p2_1k8:
                    if p2 > p1 and p2 % 16 == 9:
                        leg2 = []
                        for a in a_base:
                            leg2.append(numth.jacobi(a, p2))
                        if np.prod(leg2) == 1 and p2 > p1:  # если все 1, то произведение 1
                            p_exist = np.array([p1, p2])
                            p3_list = next_p(p_exist, a_base)
                            if isinstance(p3_list, list):
                                for p3 in p3_list:
                                    if p3 > p2:
                                        signss = check_signs(a_base, [p1, p2, p3])
                                        if signss and p3 > p2:
                                            s += f"\np1 is 8k+5\np2 is 8k+1 or 16k+9\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                            n_list.append([p1, p2, p3])
                            else:
                                continue

                    elif p2 > p1 and p2 % 16 == 1:
                        if Mu_p(a_base, p2) == 4:
                            break  # переход к mu=4
            elif p1 % 8 == 1:
                e, f = Val(2, p1 - 1), Val(2, Lambda_p(a_base, p1))

                if len(a_base) > 5:
                    a_base = a_base[:5]
                for p2 in primes:
                    if p2 > p1 and e == f:
                        if p2 % (2 ** (e + 1)) == (1 + 2 ** e) % (2 ** (e + 1)):  # !!!! СКОБКИ???
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            if leg1 == leg2:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 > p2:
                                            signss = check_signs(a_base, [p1, p2, p3])
                                            if signss:
                                                s += f"\np1 is 8k+1\ne==f\nleg1==leg2\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                                n_list.append([p1, p2, p3])
                                else:
                                    continue


                        elif p2 % 2 ** (e + 2) == (1 + 2 ** (e + 1)) % 2 ** (e + 2):
                            leg2 = []
                            for a in a_base:
                                leg2.append(numth.jacobi(a, p2))
                            if np.prod(leg2) == 1:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 > p2:
                                            signss = check_signs(a_base, [p1, p2, p3])
                                            if signss:
                                                s += f"\np1 is 8k+1\ne==f\nleg2==1\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                                n_list.append([p1, p2, p3])
                                else:
                                    continue


                        elif p2 % (2 ** (e + 2)) == 1:
                            if Mu_p(a_base, p2) == 4:
                                break  # переход к mu=4

                    elif p2 > p1 and f < e:
                        if p2 % 2 ** f == p1:
                            if f == e - 1 and Mu_p(a_base, p2) == 2:
                                p_exist = np.array([p1, p2])
                                p3_list = next_p(p_exist, a_base)
                                if isinstance(p3_list, list):
                                    for p3 in p3_list:
                                        if p3 > p2:
                                            signss = check_signs(a_base, [p1, p2, p3])
                                            if signss:
                                                s += f"\np1 is 8k+1\nf<e\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                                n_list.append([p1, p2, p3])
                                else:
                                    continue

            # p1 is any
            for p2 in mu_4:  # если p2 mu=4 то не обязательно чтобы и p1 mu=4, главное найти одинаковую сигнатуру!!!
                if p2 > p1:
                    p_exist = np.array([p1, p2])
                    p3_list = next_p(p_exist, a_base)
                    if isinstance(p3_list, list):
                        for p3 in p3_list:
                            if p3 > p2:
                                signss = check_signs(a_base, [p1, p2, p3])
                                if signss:
                                    s += f"\np1 is 8k+1\nf<e\n{p1}   {p2}   {p3}    {signss} {Sign(a_base,p1)}\n"
                                    n_list.append([p1, p2, p3])
                    else:
                        continue


            writefile(f"res/j'n'd/3/n_list_{a_base}_3_{primes_list[0]}.txt", s)
        else:
            print(f"Value Error: p1 > {int(root(Q11, 2))}\n")
            break

    i = 0
    spsp = []
    ss = ""
    for turple in n_list:
        prod = np.prod(turple)
        if psp(a_base, prod):
            ss += f"{i}    {prod}  {turple}    {Sign(a_base,turple[0])}\n"
            i += 1
            spsp.append(turple)
    ###
    total_time = "--- %s seconds ---\n" % (time.time() - start_time)
    ###

    ss += f"{total_time}\n"
    writefile(f"res/j'n'd/3/spsp_{a_base}_3_{primes_list[0]}.txt", ss)
    return np.array(spsp)


def step_t_4(a_base, p1):
    if p1 <= int(root(Q11, 4)):
        ### Посчет времени работы
        start_time = time.time()
        ###
        n_list = []
        p_list = step_t_3(a_base, p1)
        for primes_3 in p_list:

            if p1 % 4 == 3:
                n_list = [p1]
                q_3k4 = readfile("primes/4k+3.txt")
                sign_p1 = []
                for a in a_base:
                    sign_p1.append(numth.jacobi(a, p1))

                for q in q_3k4:
                    if q <= Q11 / np.prod(primes_3):
                        sign_q = []
                        for a in a_base:
                            sign_q.append(numth.jacobi(a, q))
                        if sign_q == sign_p1:
                            n_list.append(q)

            else:
                p4_list = next_p(primes_3, a_base)
                for p4 in p4_list:
                    n_list.append(primes_3 + [p4])

        ###
        total_time = "--- %s seconds ---\n" % (time.time() - start_time)
        ###
        return np.array(n_list)
    else:
        print(f"Value Error: p1 > {int(root(Q11, 4))}")


def step_t_5(a_base, p1):
    if p1 <= int(root(Q11, 5)):
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
        print(f"Value Error: p1 > {int(root(Q11, 5))}")


if __name__ == "__main__":
    print(sys.maxsize)
    # print(sys.version)

    # step_t_2(bases[:5], primes[25:1229])  # 10**2..10**5
    # step_t_2(bases[:5], primes[1229:9592])  # 10**5..10**6
    # step_t_2(bases[:2], primes[9592:78498])  # 10**6..10**7
    # step_t_2(bases[:3], primes[78498:664579])#10**7..10**8
    # step_t_2(bases[:3], primes[664579:])#10**8..15*10**6
    # step1(bases[:3])
    # step1(bases[:7])

    step_t_3(bases[:3], primes[25:1229])  # 10**2..10**5
    # step_t_3(bases[:5], primes[1229:9592])  # 10**5..10**6
