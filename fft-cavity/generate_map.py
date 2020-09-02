import numpy as np
import scipy.fftpack as sf


def generate_map(N, c_l):
    y = np.random.randn(N)
    x = np.linspace(-0.5, 0.5, N)
    F = np.exp(-0.5 * x**2 / c_l**2)
    res = (sf.ifft(sf.fft(F) * sf.fft(y))).real

    res -= res.min()
    res /= res.max()
    res -= 0.5
    return res


def generate_map2d(dim, c_l):
    s = np.random.randn(dim[0], dim[1])
    x = np.linspace(-0.5, 0.5, dim[0])
    y = np.linspace(-0.5, 0.5, dim[1])
    X, Y = np.meshgrid(x, y)

    F = np.exp(-0.5 * (X**2 + Y**2) / c_l**2)
    res = sf.ifft2(sf.fft2(F) * sf.fft2(s)).real

    res -= res.min()
    res /= res.max()
    res -= 0.5
    return res


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    n = 2**9
    c_l = 0.1

    m = generate_map2d((n, n), c_l)
    m1D = generate_map(n, c_l)

    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(m)
    axes[1].plot(m1D)
    plt.show()
