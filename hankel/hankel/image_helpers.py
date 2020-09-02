import math

import numpy as np
import scipy.interpolate as si


def image_from_radial(profile, N, ratio=1.0):
    r = np.arange(len(profile))

    fun = si.interp1d(r, profile, bounds_error='fill', fill_value=0)

    x = np.linspace(-len(profile)//2, len(profile)//2, N)
    X, Y = np.meshgrid(x, x)
    R = np.sqrt(X**2 + Y**2) * ratio

    return fun(R)

def stack_images(images, n_per_row=5):
    lines = []
    for p in range(math.ceil(len(images) / n_per_row)):
        images_in_row = images[p * n_per_row:(p + 1) * n_per_row]
        if len(images_in_row) < n_per_row:
            images_in_row += [np.zeros(images_in_row[0].shape)
                              for k in range(n_per_row - len(images_in_row))]
        lines.append(np.concatenate(images_in_row, axis=1))
    return np.concatenate(lines)
