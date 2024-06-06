"""
DEIM algorithm for determining the best columns of A.
Defined in A DEIM Induced CUR Factorization by
D. C. Sorensen AND M. Embree https://arxiv.org/pdf/1407.5516
"""

import numpy as np
import math

def DEIM(A: np.matrix, k: int) -> list[int]:
    m = np.shape(A)[0]
    # Extract the first k columns of A
    A_k = A[:, :k]

    # Define list of column indicies (matrix should grow to be of length k)
    p = list()

    # Set the first element of p to be indicie of max element A_1
    p_0 = np.argmax(abs(A_k[:, 0]))

    # Save p_0 to p
    p.append(p_0)

    for j in range(1, k):
        # a = A_k(:, j)
        a = A_k[:, j : j + 1]

        # c = A_k(p, 1 : j − 1)^(−1) a(p)
        # need to select the rows of A corresponding to the indicies in p
        # then invert that submatrix and multiply it by a(p)
        subA_k = A_k[p, :j]
        invSubA_k = np.linalg.inv(subA_k)
        c = np.dot(invSubA_k, a[p])

        # r = a − A_k(:, 1 : j − 1)c
        r = a - np.dot(A_k[:, :j], c)

        # [∼, p_j ] = max(|r|)
        p_j = np.argmax(abs(r))

        # p = [p; p_j]
        p.append(p_j)
    return p
