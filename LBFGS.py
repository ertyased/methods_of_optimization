import numpy as np
from collections import deque
from random import randint

class Myfunction:

    def __init__(self, N):
        self.N = N

    def func(self, values):
        values = np.transpose(values)[0]
        ans = 0
        for i in range(self.N // 2):
            i_2 = float(values[i * 2])
            i_2_1 = float(values[i * 2 + 1])
            ans += 100 * (i_2 ** 2 - i_2_1) ** 2 + (i_2 - 1) ** 2
        return ans

    def gradient(self, values):
        values = np.transpose(values)[0]
        gradient = []
        for i in range(self.N // 2):
            x = float(values[i * 2])
            y = float(values[i * 2 + 1])
            gradient.append(-2 + 2 * x - 400 * y * x + 400 * x ** 3)
            gradient.append(200 * (y - x ** 2))
        return np.array(gradient)


def L_BFGS(start, m, object, limit=100_000, threshold=1e-4):
    print(start)
    s = deque()
    y = deque()
    rho = deque()
    start = np.transpose([start])
    prev_x = -1
    prev_g = -1
    for k in range(limit):
        g_k = np.transpose([object.gradient(start)])
        if np.linalg.norm(g_k) < threshold:
            print(f"found in {k} steps")
            return start
        x_k = start
        if k != 0:
            s.append(x_k - prev_x)
            y.append(g_k - prev_g)
            rho.append(1 / np.matmul(np.transpose(y[-1]), s[-1])[0][0])
            if k > m:
                s.popleft()
                y.popleft()
                rho.popleft()
        q = g_k
        prev_x = x_k
        prev_g = g_k
        alphas = [0] * len(s)
        for i in range(1, len(s) + 1):
            alpha = rho[-i] * np.matmul(np.transpose(s[-i]), q)[0][0]
            q = q - alpha * y[-i]
            alphas[-i] = alpha

        gamma = 1
        if k != 0:
            gamma = np.matmul(np.transpose(s[-1]), y[-1]) / np.matmul(np.transpose(y[-1]), y[-1])

        z = q * gamma

        for i in range(min(k, m)):
            beta = rho[i] * np.matmul(np.transpose(y[i]), z)
            z = z + s[i] * (alphas[i] - beta)
        start = x_k - z
    return start




def test():
    N = [2, 4, 8, 10, 20, 100, 150]
    for i in N:
        print("testing for N: ", i)
        a = Myfunction(i)
        print(np.transpose(L_BFGS([randint(-10, 10) for j in range(i)], 10, a))[0])


if __name__ == "__main__":
    test()
