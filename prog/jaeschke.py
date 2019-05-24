from gmpy2 import gcd

import ecdsa.numbertheory as numth

sieve_base = [2, 3, 5, 7, 11, 13, 17, 19,
              23, 29, 31, 37, 41, 43, 47, 53]


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
    sgm_v_p = []
    for a in v:
        if gcd(int(a), int(p)) == 1:
            ord = Ord(p, a)
            val = Val(2, ord)
            # print(a, p, ord, val)
            sgm_v_p.append(val)
    return sgm_v_p


# вычисление сигнатуры для 4k+3
def Sign2(v, p):
    sgm_v_p = []
    for a in v:
        if gcd(int(a), int(p)) == 1:
            if numth.jacobi(a, p) == 0:
                sgm_v_p.append(0)
            elif numth.jacobi(a, p) == -1:
                sgm_v_p.append(1)
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


if __name__ == "__main__":
    print(sieve_base[:9], 16985)
