import numpy as np

def readFingerprints(file: str) -> list[np.matrix]:
    info = np.genfromtxt(file,max_rows=1,dtype=str,delimiter=";")
    A = np.genfromtxt(file,skip_header=1,dtype=float)
    return [info,A]