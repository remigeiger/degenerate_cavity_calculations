import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate as si


def image_from_profile(x_profile, y_profile, N=100, scale=1.0):
    x = x_profile[len(x_profile) // 2:]
    y = y_profile[len(y_profile) // 2:]

    fun = si.interp1d(x, y, 'cubic', bounds_error=False,
                      fill_value=(y[0], 0))

    u = v = np.linspace(-x[-1] * scale, x[-1] * scale, N)
    U, V = np.meshgrid(u, v)

    return fun(np.sqrt(U**2 + V**2))


if __name__ == '__main__':
    x = np.linspace(-10, 10, 1000)
    E = np.exp(-x**2)

    fig, ax = plt.subplots()
    img = image_from_profile(x, E, scale=0.1)
    ax.imshow(img)
    plt.show()
