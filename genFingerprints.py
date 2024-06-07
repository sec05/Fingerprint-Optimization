import numpy as np

def genFingerprints(m: int, n: int) -> np.matrix:
    A = np.random.rand(m,n)
    A*=100
    return A

if __name__ == "__main__":
    genFingerprints(5,5*10**6)

    
