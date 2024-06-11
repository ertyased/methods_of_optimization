from SimplexMethod import *

def remake_for_binary(A):
    m = len(A) - 1
    new_A = []
    n = len(A[0]) - m - 1
    for i in range(len(A) - 1):
        add = [0] * (m + n)
        add[i] = 1
        new_A.append(A[i][:n] + add + [A[i][-1]])
    for i in range(n):
        new_l = [0] * (len(new_A[0]))
        new_l[i] = 1
        new_l[i + n + m] = 1
        new_l[-1] = 1
        new_A.append(new_l)
    new_A.append(A[-1][:n] + [0] * (m + n) + [A[-1][-1]])
    return new_A


def gomori_cut(A, result, basis):
    m = len(A) - 1
    n = len(A[0]) - m - 1
    maxin = -1
    for i in range(n):
        if maxin == -1 and abs(result[i] - round(result[i])) > 1e-5:
            maxin = i
        else:
            continue
        if result[maxin] % 1 <= result[i] % 1:
            maxin = i
    if maxin == -1:
        return A, result, basis
    row = 0
    for i in range(len(basis)):
        if maxin == basis[i]:
            row = i
    new_line = []
    for j in range(len(A[row]) - 1):
        i = A[row][j]
        new_line.append(-(i % 1))
    new_line.append(1)
    new_line.append(-(result[maxin] % 1))


    new_A = []
    for i in range(len(A) - 1):
        b = A[i]
        new_A.append(b[:-1] + [0] + [b[-1]])
    new_A.append(new_line)
    new_A.append(A[-1][:-1] + [0] + [A[-1][-1]])

    inp = -1
    minim = -10**18
    for i in range(len(new_A[0]) - 1):
        if abs(new_A[-2][i]) > EPS and i != n + m:
            if minim < new_A[-1][i] / new_A[-2][i]:
                minim = new_A[-1][i] / new_A[-2][i]
                inp = i

    basis.append(inp)
    new_A = make_iteration(new_A, inp, m)
    values = get_result(new_A, basis)
    return gomori_cut(new_A, values, basis)


if __name__ == "__main__":
    n, A = read()

    print("Minimum of:", to_string(list(map(lambda x: -x, A[-1][:n]))))
    print("With conditions:")
    for i in range(len(A) - 1):
        print(to_string(A[i][:n]), "<=", A[i][-1])
    print("and xi are boolean")
    new_A = remake_for_binary(A)


    res_A, basis = simplex(new_A, get_inp_out, [n + i for i in range(len(new_A) - 1)])
    res_A[-1] = [-i for i in res_A[-1]]
    values = get_result(res_A, basis)
    res_A, result, basis = gomori_cut(res_A, values, basis)
    print("is:", -res_A[-1][-1])
    print("At: ", end="")
    for i in range(n):
        print(f"x{i + 1}={round(result[i])} ", end=" ")

""""
Примеры входных данных:

3 2
-3 -2 -4
1 1 1 2
5 0 6 5
"""
