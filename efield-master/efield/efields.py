from efield import EField
import numpy as np
import math


def laguerre(n, X):
    if n == 0:
        return 1 + 0 * X
    elif n == 1:
        return 1 - X
    else:
        lg0 = 1
        lg1 = 1 - X
        for k in range(2, n + 1):
            lg = ((2 * k - 1 - X) * lg1 - (k - 1) * lg0) / k
            lg0, lg1 = lg1, lg
        return lg


def hermite(n, X):
    if n == 0:
        return 1 + 0 * X
    elif n == 1:
        return 2 * X
    else:
        H0, H1 = 1, 2 * X
        for k in range(2, n + 1):
            H = 2 * X * H1 - 2 * (k - 1) * H0
            H1, H0 = H, H1
        return H


def hg_efield(n, x, w, amplitude=1.0, x0=0.0, normalize_power=True, *args, **kwargs):
    if normalize_power:
        factor = np.sqrt(2**(1 - n) / np.pi / math.factorial(n)) / w
    else:
        factor = amplitude
    field = hermite(n, np.sqrt(2) * (x - x0) / w) * np.exp(-(x - x0)**2 / w**2)
    return EField(x=x, E=factor * field, *args, **kwargs)


def lg_efield(n, x, w, amplitude=1.0, x0=0.0, *args, **kwargs):
    factor = np.sqrt(2**(1 - n) / np.pi / math.factorial(n)) / w
    field = laguerre(n, 2 * abs(x - x0)**2 / w**2) * np.exp(-(x - x0)**2 / w**2)
    return EField(x=x, E=factor * field, *args, **kwargs)


def fermidirac(x, amplitude, R0, x0, b):
    e = np.exp(-b * (abs(x - x0) / R0 - 1))
    return amplitude * e / (1 + e)


def fd_efield(x, R0, b, amplitude=1.0, x0=0.0, *args, **kwargs):
    return EField(x=x,
                  E=fermidirac(x, amplitude, R0, x0, b),
                  *args, **kwargs)


def example():

    plt.subplots()
    w = 0.5e-3
    x0 = 5 * w
    N = 2**12
    x = np.linspace(-x0, x0, N)
    for n in range(3):
        E = lg_efield(n=n, x=x, w=w, amplitude=1.0, x0=0.0, radial=True, normalize_power=True)
        E = hg_efield(n=n, x=x, w=w, radial=True, normalize_power=True)
        axes = plt.plot(x / w, E.I, label=f'n : {n}')

        plt.plot(x / w, E.propagated(0.4).I, linestyle='--', color=axes[-1].get_c())
        print(f'{n} : {E.P}')

    plt.legend()
    plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    example()
