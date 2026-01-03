from pprint import pprint

from suitesparse_amd import amd
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    n = 10

    a = np.eye(n)
    a[-1][0] = 1
    a[0][-1] = 1
    a[0, 0] = 2

    sym = a
    # sym = a @ a.T + np.eye(n)
    print(sym)

    P, info = amd.amd(sym.tolist(), verbose=True, aggressive=True, dense=10.0)

    print(info)

    print(P)
    full_P = np.zeros((n, n))

    full_P[np.arange(n), P] = 1

    print(full_P)

    Lbase = np.linalg.cholesky(sym)
    pprint(np.around(Lbase, 1))

    modified_sym = full_P @ sym @ full_P.T

    Lnew = np.linalg.cholesky(modified_sym)

    pprint(np.around(Lnew, 1))

    fig, axes = plt.subplots(2, 2, figsize=(5,5))

    axes[0][0].set_title("Original")
    axes[0][0].imshow(sym)
    axes[0][1].set_title("Ordered")
    axes[0][1].imshow(modified_sym)

    mask = Lbase == 0
    mask_n = Lnew == 0

    axes[1][0].imshow(Lbase != 0)
    axes[1][1].imshow(Lnew != 0)

    plt.tight_layout()

    plt.savefig("demo.png")

    print("Number of zeros:")
    print("Previous: ", mask.sum(), "New:", mask_n.sum())
