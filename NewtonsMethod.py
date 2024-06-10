import numpy as np
from random import random

class Myfunction:
    def func(self, z):
        x, y, lambda_ = z
        return (1 - x)**2 + 100 * (y - x**2)**2 - lambda_ * (y + x**2)

    def gradient(self, z):
        x, y, lambda_ = z.tolist()
        return np.array([
            -2 + 2 * x - 400 * y * x + 400 * x**3 - 2 * lambda_ * x,
            200 * (y - x**2) - lambda_,
            -y - x**2
        ])

    def jacobian(self, z):
        x, y, lambda_ = z
        return np.array([
            [2 - 400 * y + 1200 * x**2 - 2 * lambda_, -400 * x, -2 * x],
            [-400 * x, 200, -1],
            [-2 * x, -1, 0]
        ])


def Newton_method(x0, object: Myfunction, steps_amount=10000):
    x_k = x0
    for _ in range(steps_amount):
        x_k = x_k - np.matmul(np.linalg.inv(object.jacobian(x_k)), object.gradient(x_k)) # так как матрица маленькая
    return x_k # не страшно что ищем обратную к ней.


if __name__ == "__main__":
    amount_tests = 10
    for i in range(amount_tests):
        x = random() * 5
        lambda_ = random() * 100
        print(f"run {i}. For starting points x={x}, y={-x * x}, lambda={lambda_}")
        x0 = np.array([x, - x * x, lambda_])
        x, y, lambda_ = Newton_method(x0, Myfunction())

        print(f"result: x={x}, y={y}, lambda={lambda_}\n."
             f" difference with constraint {abs(y + x * x)}")

"""
    Программа успешно находит минимум функции с заданным ограничением при старте с случайной точки.
"""
