from jiangdeng import step_t_2
from utils import readfile

bases = [2, 3, 5, 7, 11, 13, 17, 19,
         23, 29, 31, 37, 41, 43, 47, 53]

primes = readfile("primes/primes_1m.txt")

if __name__ == "__main__":
    for i in range(6, 12, 2):
        for j in range(2, 5):
            print(i, j)
            step_t_2(bases[:j], 10 ** i, primes)
