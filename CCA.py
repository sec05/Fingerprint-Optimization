'''
This file takes in a matrix A and int k and returns an approximation B of A where A ~ B = CC^+A.
C is made up of k columns of A and we just return C
'''

import numpy as np
from DEIM import DEIM

def CCA(A: np.matrix, k: int) -> np.matrix:
    
    # Compute SVD (we only need V)
    _, _, V = np.linalg.svd(A)
    V = V.transpose()

    # Apply DEIM to V
    qs = DEIM(V,k)

    # Form C
    C = A[:,qs]

    return C

