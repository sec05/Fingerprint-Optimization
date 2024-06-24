import numpy as np
def main():
    A = np.genfromtxt("Matrix Output/A.matrix",dtype=float)
    B = np.genfromtxt("Matrix Output/vals.matrix")
    C = np.genfromtxt("Matrix Output/vecs.matrix")
    vals, vecs = np.linalg.eigh(A)
    print("Answers:\n",vals,"\n",vecs)
    print("Outputs:\n",B,"\n",C)

if __name__ == "__main__": main()