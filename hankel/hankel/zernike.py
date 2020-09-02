import numpy as np


def zernike(n, m, c, u, phi=0.0):
    if isinstance(phi, float):
        phi = np.ones(u.shape)*phi
    phi[u < 0] += np.pi
    u = abs(u)
    if (n, m) == (0, 0):
        out = c*np.ones(u.shape)
    elif (n, m) == (1, 1):
        out = c * 2*u * np.cos(phi)
    elif (n, m) == (1, -1):
        out = c * 2*u * np.sin(phi)
    elif (n, m) == (2, 0):
        out = c * np.sqrt(3)*(2*u**2 - 1)
    elif (n, m) == (2, 2):
        out = c * np.sqrt(6) * u**2 * np.cos(2*phi)
    elif (n, m) == (2, -2):
        out = c * np.sqrt(6) * u**2 * np.sin(2*phi)
    elif (n, m) == (3, 1):
        out = c * np.sqrt(8) * (3*u**3 - 2*u) * np.cos(phi)
    elif (n, m) == (3, -1):
        out = c * np.sqrt(8) * (3*u**3 - 2*u) * np.sin(phi)
    elif (n, m) == (4, 0):
        out = c * np.sqrt(5) * (6*u**4 - 6*u**2 + 1)
    else:
        raise ValueError('({}, {}) is not implemented'.format(n, m))
    out[u > 1] = 0
    non_zeros = np.nonzero(out)[0]
    if len(non_zeros) > 1:
        out[:non_zeros[0]] = out[non_zeros[0]]
        out[non_zeros[-1]:] = out[non_zeros[-1]]
    return out


def zernike_combination(orders, coeffs, u, phi):
    out = np.zeros(u.shape)
    for (n, m), c in zip(orders, coeffs):
        out += zernike(n, m, c, u, phi)
    return out


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    u = np.linspace(-2, 2, 100)
    phase = zernike_combination([(0, 0), (1, -1), (1, 1), (2, 0),
                                 (2, -2), (2, 2), (3, -1), (3, 1), (4, 0)],
                                [0, 0, 0, 0,
                                 0.012, -0.011, 0.072, 0.040, 0.109],
                                u, phi=1.5*np.pi/2.0)

    plt.plot(u, phase)
    plt.show()
