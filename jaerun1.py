from jaeschke import t_2, t_more_3
from utils import readfile

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

primes = readfile("primes/primes_1m.txt")

if __name__ == "__main__":
    print(2, 6)
    t_more_3(bases[:2], 10 ** 6, primes)

