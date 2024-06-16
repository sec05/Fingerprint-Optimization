from CUR import CUR
from CCA import CCA
#from genFingerprints import genFingerprints
import numpy as np
import matplotlib.pyplot as plt
import time
from plotMatrixSizeVsError import plotMSVE
from plotMatrixSizeVsTime  import plotMSVT
def main():
    ts = []
    es = []
    ks = []
    for k in range (1,11):
        ks.append(k)
        n=2**k
        A = np.random.rand(n,n)

        start = time.time()
        C = CCA(A,k)
        end = time.time()
        ts.append(end-start)

        Cp = np.linalg.pinv(C)

        e = np.linalg.norm(A - (C @ Cp @ A), 'fro')
        e /= np.linalg.norm(A,'fro')
        es.append(e)
    plotMSVE(ks,es)
    plotMSVT(ks,ts,True)
    
if __name__ == "__main__":
    main()
