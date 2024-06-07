from CUR import CUR
from CCA import CCA
from genFingerprints import genFingerprints
import numpy as np

def main():
    k = 2
    A = genFingerprints(10,10**4)
    C = CCA(A,k)
    
    Cp = np.linalg.pinv(C)

    e = np.linalg.norm(A - (C @ Cp @ A), 'fro')
    e /= np.linalg.norm(A,'fro')

    print("Error is ", e)

    return 0

if __name__ == "__main__":
    main()
