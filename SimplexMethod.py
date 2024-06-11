EPS = 1e-5

def get_inp_out(A):
    b = A[-1]
    inp = 0
    for i in range(len(b)):
        if b[i] >= b[inp]:
            inp = i

    out = 0
    min_out = 10**18
    for i in range(len(A) - 1):
        c = A[i]
        if c[inp] <= 0:
            continue
        if min_out > c[-1] / c[inp]:
            out = i
            min_out = c[-1] / c[inp]
    return inp, out


def remake(a, b, inp):
    c = a[inp]
    d = [0] * len(a)
    for i in range(len(a)):
        d[i] = a[i] - c * b[i]
    d[inp] = 0
    return d


def make_iteration(A, inp, out):
    c = A[out][inp]
    for i in range(len(A[0])):
        A[out][i] /= c

    for i in range(len(A)):
        if i != out:
            A[i] = remake(A[i], A[out], inp)
    return A


def check(A):
    b = A[-1][:-1]
    for i in b:
        if i > EPS:
            return False
    return True


def get_result(A, basis):
    result = [0] * len(A[0])
    for i in range(len(basis)):
        result[basis[i]] = A[i][-1]
    return result


def simplex(A_c, inp_out, basis):
    A = []

    for i in A_c:
        A.append(i.copy())
    while not check(A):
        inp, out = inp_out(A)
        basis[out] = inp
        A = make_iteration(A, inp, out)
    return [A, basis]


def to_string(a):
    ans = ""
    for i in range(len(a)):
        if i != 0:
            if a[i] >= 0:
                ans += " + "
            if a[i] < 0:
                ans += " - "
            ans += f"{abs(a[i])}*x{i + 1}"
        else:
            ans += f"{a[i]}*x{i + 1}"
    return ans


def read():
    n, m = map(int, input().split())
    a = list(map(int, input().split()))
    A = []
    for i in range(m):
        b = list(map(int, input().split()))
        new_b = b[:-1] + [0] * m + [b[-1]]
        new_b[n + i] = 1
        A.append(new_b)
    a = list(map(lambda x: -x, a)) + [0] * m + [0]
    A.append(a)
    return n, A



if __name__ == "__main__":
    n, A = read()
    print("Minimum of:", to_string(list(map(lambda x: -x, A[-1][:n]))))
    print("With conditions:")
    for i in range(len(A) - 1):
        print(to_string(A[i][:n]), "<=", A[i][-1])
    res_A, basis = simplex(A, get_inp_out, [n + i for i in range(len(A) - 1)])
    values, minim = get_result(res_A, basis), res_A[-1][-1]

    print("is:", minim)
    print("At: ", end="")
    for i in range(n):
        print(f"x{i + 1}={values[i]} ", end=" ")

""""
Примеры входных данных:

3 3
-3 -2 -4
2 1 1 10
1 3 2 15
1 1 4 20

2 2
-1 -1
1 3 6
2 1 4

3 2
-3 -2 -4
1 1 1 2
5 0 6 4
"""