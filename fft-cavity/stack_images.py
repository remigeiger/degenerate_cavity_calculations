import numpy as np
import math

def stack_images(images, n_per_row=5):
    lines = []
    for p in range(math.ceil(len(images) / n_per_row)):
        images_in_row = images[p * n_per_row:(p + 1) * n_per_row]
        if len(images_in_row) < n_per_row:
            images_in_row += [np.zeros(images_in_row[0].shape)
                              for k in range(n_per_row - len(images_in_row))]
        lines.append(np.concatenate(images_in_row, axis=1))
    return np.concatenate(lines)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    n = 128
    x = np.linspace(-5, 5, n)

    X, Y = np.meshgrid(x, x)
    R2 = X**2 + Y**2

    gaussian = np.exp(- R2 / 2**2)

    images = [gaussian]*6

    big_img = stack_images(images, n_per_row=10)

    plt.figure()
    plt.imshow(big_img)
    plt.show()