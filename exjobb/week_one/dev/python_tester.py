import numpy as np

def test1():
    matrix = np.zeros((4,4))
    counter = 1
    for i in range(0,4):
        for j in range(0,4):
            matrix[i][j] = counter
            counter += 1

    print(matrix)

    vector = np.asarray(matrix).reshape(-1)
    print(vector)

def test2():
    n = 5
    N = n-2
    A1 = np.zeros((N**2,N**2))
    A2 = A1.copy()
    for i in range(0, N):
        A1[i * N + N - 1, i * N + N - 1] = -3
        A2[i * N, i * N] = -3
    print(A1)
    print(A2)

def test3():
    matrix = np.zeros((4,4))
    counter = 1
    for i in range(0,4):
        for j in range(0,4):
            matrix[i][j] = counter
            counter += 1
    print(matrix)
    print(matrix[:,-1])



test3()