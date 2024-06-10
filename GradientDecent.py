import numpy as np
from random import random, randint
from NewtonsMethod import Myfunction

a = Myfunction()

def gradient_norm(z0):
    grad = a.gradient(z0)
    return grad[0]**2 + grad[1]**2 + grad[2]**2

def gradient_of_gradient_norm(z0):
    grad = a.gradient(z0)
    jacob = a.jacobian(z0)

    return np.array([
        2 * grad[0] * jacob[0][0] + 2 * grad[1] * jacob[0][1] + 2 * grad[2] * jacob[0][2],
        2 * grad[0] * jacob[1][0] + 2 * grad[1] * jacob[1][1] + 2 * grad[2] * jacob[1][2],
        2 * grad[0] * jacob[2][0] + 2 * grad[1] * jacob[2][1] + 2 * grad[2] * jacob[2][2]
    ])


def backtracking_line_search(f, grad_f, x, p, alpha=1, rho=0.5, c=1e-4):
    """
    Метод обратного отслеживания для выбора шага с использованием условия Армихо.

    :param f: Целевая функция
    :param grad_f: Градиент целевой функции
    :param x: Текущая точка
    :param p: Направление спуска
    :param alpha: Начальная длина шага
    :param rho: Коэффициент уменьшения шага (обычно 0.5)
    :param c: Константа для условия Армихо (обычно 1e-4)
    :return: Оптимальная длина шага
    """
    while f(x + alpha * p) > f(x) + c * alpha * np.dot(grad_f(x), p):
        alpha *= rho
    return alpha

def choose_step(x, d):
    return backtracking_line_search(gradient_norm, gradient_of_gradient_norm, x, d)


class Adam:
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def update(self, x, grad_x):
        if self.m is None:
            self.m = np.zeros_like(grad_x)
        if self.v is None:
            self.v = np.zeros_like(grad_x)

        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad_x
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad_x**2

        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)

        adjusted_grad = m_hat / (np.sqrt(v_hat) + self.epsilon)
        return x - self.learning_rate * adjusted_grad


def gradientDecent(start, threshold=1e-6, limit=100_000):
    i = 0
    adam = Adam()
    while True:
        i += 1
        grad = gradient_of_gradient_norm(start)
        if np.linalg.norm(grad) < threshold or i == limit:
            print(f"found in {i} steps")
            return start
        start = adam.update(start, grad)


if __name__ == "__main__":
    amount_tests = 1
    for i in range(amount_tests):
        starting_points = [
            [0.3, 0.5, -10],
            [0.2, 0.1, 0.3],
            [0.1038554, -0.010785957, -4.314383],
            [1, 1, 1]
        ]
        for test in starting_points:
            x0 = np.array(test)

            print(f"run {i}. For starting points x={x0[0]}, y={x0[1]}, lambda={x0[2]}")

            x, y, lambda_ = gradientDecent(x0)

            print(f"result: x={x}, y={y}, lambda={lambda_}."
                  f"difference with constraint {abs(y + x * x)}")

            print(f"value is {a.func([x, y, lambda_])}")


"""
    Градиентный спуск с трудом справляется c поиском минимума Лагранжиана, даже при поиске минимума нормы градиенты.
    Пришлось попробовать несколько методов выбора шага. Adam справился лучше всех.
"""
