import numpy as np
def main():
    A = np.genfromtxt("Matrix Output/randomSymmetric.matrix",dtype=float)
    B = np.genfromtxt("Matrix Output/tridiag.matrix")
    print("Det A =",np.linalg.det(A))
    print("Det B=",np.linalg.det(B))
    m = A.shape[0]
    for k in range(m-2):
        # Create the vector x
        x = A[k+1:, k]

        # Calculate the norm of x and e1
        e1 = np.zeros_like(x)
        e1[0] = np.linalg.norm(x) * np.sign(x[0]) if x[0] != 0 else np.linalg.norm(x)

        # Create the Householder vector v
        v = x + e1
        v /= np.linalg.norm(v)

        # Update the matrix A
        # Compute the Householder transformation P = I - 2 * v * v^T
        P = np.eye(m-k-1) - 2.0 * np.outer(v, v)
        
        # Apply P from the left and right to A[k+1:m, k:m]
        A[k+1:, k:] = P @ A[k+1:, k:]
        A[:, k+1:] = A[:, k+1:] @ P.T
    print("Det T=",np.linalg.det(A))
    np.savetxt("./Matrix Output/answer.matrix",A)
    return 0

if __name__ == "__main__": main()