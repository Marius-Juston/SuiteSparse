from pprint import pprint

import _amd
import numpy as np

if __name__ == '__main__':
    n = 100

    a = np.random.randint(2, size=(n, n))

    sym = a @ a.T + np.eye(n)
    print(sym)

    P = _amd.amd(sym)

    print(P)
    full_P = np.zeros((n, n))

    full_P[np.arange(n), P] = 1

    print(full_P)

    Lbase = np.linalg.cholesky(sym)
    pprint(np.around(Lbase, 1))

    Lnew = np.linalg.cholesky(full_P @ sym @ full_P.T)

    pprint(np.around(Lnew, 1))
