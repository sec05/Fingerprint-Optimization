'''
This file takes in a matrix A and int k and returns an approximation B of A where A ~ B = CC^+A.
C is made up of k columns of A and we just return C
'''

import numpy as np
import scipy
import scipy.linalg
from DEIMParallel import DEIMParallel

def CCA(A: np.matrix, k: int) -> np.matrix:
    
    # Compute eigenvectors of A^TA
    A = A.transpose() @ A
    _,v = scipy.linalg.eigh(A)
    V=np.matrix(v)
    # Apply DEIM to V
    qs = DEIMParallel(V,k)
    # Form C
    C = A[:,qs]

    return C

