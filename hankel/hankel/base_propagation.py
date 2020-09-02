from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ss

import hankel


def main():
    N = 2**12
    a = 7e-3

    rs, ps = hankel.rp_space(N, a)
    Y = hankel.kernel(N)

    w = a/10
    lda = 852e-9
    zr = np.pi * w**2 / lda
    E = np.exp(-rs**2/w**2)

    d = 1

    phase_shift = np.pi * lda * d * ps**2

    t = time()
    E_prop = Y.dot(Y.dot(E) * np.exp(-1j*phase_shift))
    print(time() - t)

    P = ss.diags(np.exp(-1j*phase_shift))
    M = Y.dot(P.dot(Y))

    t = time()
    E_prop2 = M.dot(E)
    print(time() - t)

    w_th = w * np.sqrt(1 + (d/zr)**2)
    E_th = w / w_th * np.exp(-rs**2/w_th**2)
    bt = Y.dot(E)

    plt.figure()
    plt.plot(ps, abs(bt))

    plt.figure()
    plt.plot(rs, abs(E)**2)
    plt.plot(rs, abs(E_prop)**2)
    plt.plot(rs, abs(E_prop2)**2)
    plt.plot(rs, abs(E_th)**2, linestyle='--')

    plt.show()


if __name__ == '__main__':
    main()