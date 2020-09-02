from time import time

import matplotlib.pyplot as plt
import numpy as np

import hankel


def main():
    N = 2**7

    if N**2 * 128/8 * 1e-6 > 2000:
        raise RuntimeError('Wow wow wow wow..')

    a = 7e-3

    rs, ps = hankel.rp_space(N, a)
    Y = hankel.kernel(N)

    w = a/3
    lda = 852e-9
    zr = np.pi * w**2 / lda
    E = np.exp(-rs**2/w**2, dtype=np.complex128)

    f = 0.2

    dx = a/N
    print('Condition lentille a 2w : {:.2e} > 1'.format(N * lda * f / (2 * a ** 2)))
    print('Condition longue propagation : {:.2e} > 1'.format(N * dx * np.sqrt(4 * dx ** 2 - lda ** 2) / 2 / lda / f))

    phase_shift_prop = np.pi * lda * f * ps**2
    phase_shift_lens = 2*np.pi/lda * f * (1 + 0.5*rs**2 / f**2)
    
    def propagate(E, d):
        wf = np.pi * lda * d * ps**2
        return Y.dot(Y.dot(E) * np.exp(-1j*wf))
    
    def lens(E, f):
        wf = 2*np.pi/lda * f * (1 + 0.5*rs**2 / f**2)
        return E * np.exp(-1j*wf)
    

    t = time()
    E1 = propagate(E, f)
    E2 = lens(E1, f)
    E3 = propagate(E2, f)
    E4 = propagate(E3, f)
    E5 = lens(E4, f)
    E_final = propagate(E5, f)
    print(time() - t)

    plt.figure()
    plt.plot(rs, abs(E)**2)
    plt.plot(rs, abs(E_final)**2)

    plt.show()


if __name__ == '__main__':
    main()