import matplotlib.pyplot as plt
import numpy as np

def plotMSVE(size: list[int], error: list[float], fit: bool = False, degree: int = 3) -> None:
    if fit:
        coefficients =  np.polyfit(size,error, degree)
        p = np.poly1d(coefficients)
        xs = np.linspace(size[0],size[-1],10000)
        ys = p(xs)
        plt.plot(xs,ys,label=f'Degree {degree} polynomial fit',color="red")
    
    plt.scatter(size,error,label="Data",color="blue")
    plt.title("Matrix size vs error")
    plt.xlabel("Matrix size ($2^k$ x $2^k$)")
    plt.ylabel("Error")
    plt.savefig("matrix_size_vs_error.png")
    plt.close()