'''
This file takes in a matrix A and returns an approximation B of A where A ~ B = CUR.
C is made up of k columns of A, R is made up of k rows of A, and U is a 
possibly ill-conditioned matrix that optimizes the factorization.
CUR is returned as an array of the form [C, U, R].
EF = True will return [C, U, R, E, F] where
C = E^TA and R = AF
'''
import numpy as np
from DEIM import DEIM
def CUR(A: np.matrix, k: int, EF = False) -> list[np.matrix]:
    m = np.shape(A)[0]
    n = np.shape(A)[1]

    # First we need to create an SVD of our matrix
    U, _, V = np.linalg.svd(A)

    # Get p's which are row indicies
    ps = DEIM(U,k)

    # Get q's which are column indicies
    qs = DEIM(V.transpose(),k)

    # Define matrix E = [e_ps[0],...,e_ps[k]]
    E = np.zeros((m,k))
    for i in range(k):
        E[ps[i],i] = 1
    
    # Define matrix F = [f_qs[0],...,f_qs[k]]
    F = np.zeros((n,k))
    for i in range(k):
        F[qs[i],i] = 1

    # Compute R
    R = E.transpose() @ A
    
    # Compute C
    C = A @ F

    # Compute U
    Cps = np.linalg.pinv(C)
    Rps = np.linalg.pinv(R)
    U = Cps @ A @ Rps
    if EF:
        return [C, U, R, E, F]
    else:
        return [C, U, R]





