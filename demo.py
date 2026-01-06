# type: ignore

"""
Demo code for the AMD
"""

from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np

from suitesparse_amd import amd


def sparse_dense(n, density=0.15, seed=42):
    """
    Generates a random sparse matrix with a specific density
    :param n: the matrix dimension
    :param density: the desired density
    :param seed: the random seed
    :return: the sparse dense matrix
    """
    np.random.seed(seed)
    mask = np.abs(np.random.randn(n, n)) < density
    mask = np.triu(mask)
    sym_mask = mask + mask.T

    sym_mask = sym_mask.astype(np.int32)

    sym_mask -= np.diag(np.diag(sym_mask))

    sym_mask += np.diag(sym_mask.sum(axis=0) + 1)

    return sym_mask


def main():
    """
    Main demo entry point
    """
    n = 100

    a = sparse_dense(n)

    print(a)

    sym = a
    # sym = a @ a.T + np.eye(n)
    print(sym)

    permutation, info = amd.amd(sym.tolist(), verbose=True, aggressive=True, dense=10.0)

    print(info)

    print(permutation)
    full_permutation = np.zeros((n, n))

    full_permutation[np.arange(n), permutation] = 1

    print(full_permutation)

    l_base = np.linalg.cholesky(sym)
    pprint(np.around(l_base, 1))

    modified_sym = full_permutation @ sym @ full_permutation.T

    l_new = np.linalg.cholesky(modified_sym)

    pprint(np.around(l_new, 1))

    _, axes = plt.subplots(2, 2, figsize=(5, 5))

    axes[0][0].set_title("Original")
    axes[0][0].imshow(sym)
    axes[0][1].set_title("Ordered")
    axes[0][1].imshow(modified_sym)

    mask = l_base == 0
    mask_n = l_new == 0

    axes[1][0].imshow(l_base != 0)
    axes[1][1].imshow(l_new != 0)

    plt.tight_layout()

    plt.savefig("demo.png")

    print("Number of zeros:")
    print("Previous: ", mask.sum(), "New:", mask_n.sum())


if __name__ == '__main__':
    main()
