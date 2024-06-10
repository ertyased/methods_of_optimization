import numpy as np
import scipy

class Myfunction:
    def func(self, z):
        x, y = z
        return (1 - x)**2 + 100 * (y - x**2)**2

    def gradient(self, z):
        x, y = z
        return np.array([
            -2 + 2 * x - 400 * y * x + 400 * x**3,
            200 * (y - x**2)
        ])

def projection(kernel, vector):
    result = np.zeros(vector.shape)
    for i in np.transpose(kernel):
        result += np.dot(i, vector) / np.dot(i, i) * i
    return result


def linear_gradient_decent(start, kernel, object: Myfunction, step=1e-4, threshold=1e-4, limit=100_000):
    i = 0
    while True:
        i += 1
        grad = object.gradient(start)
        if np.linalg.norm(grad) < threshold or i == limit:
            print(f"found in {i} steps")
            return start
        start -= step * projection(kernel, grad)


if __name__ == "__main__":
    A = [[-1, 1]]
    b = [3]
    start = scipy.linalg.lstsq(A, b)[0]
    kernel = scipy.linalg.null_space(A)
    result = linear_gradient_decent(start, kernel, Myfunction())
    print(*result)
