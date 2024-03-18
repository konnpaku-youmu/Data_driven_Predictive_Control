import matplotlib.pyplot as plt
import numpy as np
from helper import hankelize

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 12
})

def main():
    A = np.array([[2, 3, 1],
                  [0, 1, 2]])
    
    B = np.array([[1, 3, 1],
                  [-1, 0, 1]])
    
    Π = A@B.T@np.linalg.pinv(B@B.T)@B

    ax = plt.subplot(projection = "3d")

    ax.quiver(0, 0, 0, A[0, 0], A[0, 1], A[0, 2])
    ax.quiver(0, 0, 0, A[1, 0], A[1, 1], A[1, 2])

    ax.quiver(0, 0, 0, B[0, 0], B[0, 1], B[0, 2], color="g")
    ax.quiver(0, 0, 0, B[1, 0], B[1, 1], B[1, 2], color="g")

    ax.quiver(0, 0, 0, Π[0, 0], Π[0, 1], Π[0, 2], color="r")
    ax.quiver(0, 0, 0, Π[1, 0], Π[1, 1], Π[1, 2], color="r")

    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_zlim(0, 3)
    plt.show()

    print(Π)


if __name__ == "__main__":
    main()
 