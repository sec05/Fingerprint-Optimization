from CUR import CUR
from genFingerprints import genFingerprints
import numpy as np

def main():
    k = 2
    A = genFingerprints(3,200)
    [C,U,R,E,F] = CUR(A,k,True)

    twoNorm = np.linalg.norm(A-(C@U@R),2)
    print("Error of ||A - CUR||_2 = " + str(twoNorm))

    U, S, V = np.linalg.svd(A)
    U = U[:,:k]
    V = V[:k,:]
    VtF = np.linalg.norm(np.linalg.inv(V@F),2)
    EtU = np.linalg.norm(np.linalg.inv(E.transpose()@U),2)
    e = (VtF+EtU)*S[k]
    print("Theoretical error upper bound: "+ str(e))

    return 0

if __name__ == "__main__":
    main()
