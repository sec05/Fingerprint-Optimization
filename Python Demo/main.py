from CUR import CUR
from CCA import CCA
#from genFingerprints import genFingerprints
import numpy as np
import matplotlib.pyplot as plt
import time
from plotMatrixSizeVsError import plotMSVE
from plotMatrixSizeVsTime  import plotMSVT
import csv
def main():
    with open("runtimes.csv","+a",newline='') as f:
        writer = csv.writer(f)
        for i in range(1,5):
            n = 10**i
            k = 10
            A = np.random.rand(n,n)
            s = time.time()
            CCA(A,k)
            e = time.time()
            writer.writerow(["All direct methods with scipy.linalg.eigh",n,k,e-s])
    f.close()

if __name__ == "__main__":
    main()
