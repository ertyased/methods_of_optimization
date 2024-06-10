import numpy as np
from collections import deque
from random import randint

class Myfunction:

    def __init__(self, N):
        self.N = N

    def func(self, values):
        ans = 0
        for i in range(self.N // 2):
            i_2 = values[i * 2]
            i_2_1 = values[i * 2 + 1]
            ans += 100 * (i_2 ** 2 - i_2_1) ** 2 + (i_2 - 1) ** 2
        return ans

    def gradient(self, values):
        gradient = []
        for i in range(self.N // 2):
            x = values[i * 2]
            y = values[i * 2 + 1]
            gradient.append(-2 + 2 * x - 400 * y * x + 400 * x ** 3)
            gradient.append(200 * (y - x ** 2))
        return np.array(gradient)


def wolfe_conditions(f, grad_f, x, p, alpha, c1=1e-4, c2=0.9):
    p = np.transpose(p)[0]
    x = np.transpose(x)[0]
    lhs = f(x + alpha * p)
    rhs = f(x) + c1 * alpha * np.dot(grad_f(x), p)
    if lhs > rhs:
        return False

    lhs = np.dot(grad_f(x + alpha * p), p)
    rhs= c2 * np.dot(grad_f(x), p)

    if lhs < rhs:
        return False

    return True

def choose_step(x, d, object: Myfunction):
    if wolfe_conditions(object.func, object.gradient, x, d, 1):
        return 1
    arr = [1, 2, 5]
    for i in range(1, 10):
        for j in arr:
            if wolfe_conditions(object.func, object.gradient, x, d, 10**(-i) * j):
                return 10**(-i) * j
    return -1



def L_BFGS(start, m, object, limit=100_000, threshold=1e-4):
    H_0 = np.identity(len(start))
    s = deque()
    y = deque()
    rho = deque()
    k = 0
    H_k = H_0
    start = np.transpose([start])
    for i in range(limit):
        g_k = object.gradient(start)
        x_k = start
        d_k = - np.matmul(H_k, g_k)
        if np.linalg.norm(d_k) < threshold:
            print(f"Found in {i} steps")
            return start
        step = choose_step(x_k, d_k, object)
        x_k_1 = x_k + step * d_k
        g_k_1 = object.gradient(x_k_1)
        s.append(x_k_1 - x_k)
        y.append(g_k_1 - g_k)
        rho.append(1 / np.matmul(np.transpose(y[-1]), s[-1])[0][0])
        if k >= m:
            s.popleft()
            y.popleft()
            rho.popleft()

        H_k_1 = np.zeros(H_k.shape)
        start_left = np.identity(len(start))
        for i in range(1, len(s) + 1):
            H_k_1 += rho[-i] * np.matmul(np.matmul(np.matmul(start_left, s[-i]), np.transpose(s[-i])),
                                         np.transpose(start_left))

            v_k = np.identity(len(start)) - rho[-i] * np.matmul(y[-i], np.transpose(s[-i]))

            start_left = np.matmul(start_left, np.transpose(v_k))

        H_k = H_k_1 + np.matmul(np.matmul(start_left, H_0), np.transpose(start_left))
        H_0 = H_k
        start = x_k_1
        k += 1
    print(f"Found in 100000 steps")
    return start

def test():
    N = [2, 4, 8, 20, 100]
    for i in N:
        print("testing for N: ", i)
        a = Myfunction(i)
        print(np.transpose(L_BFGS([randint(-10, 10) for j in range(i)], 10, a))[0])


if __name__ == "__main__":
    test()
