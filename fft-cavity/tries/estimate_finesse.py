import matplotlib.pyplot as plt
import numpy as np


def estimate_finesse(xaxis, spectrum, free_spectral_range=2*np.pi, peak_idx=None):
    if peak_idx is None:
        peak_idx = np.argmax(spectrum)
    peak_val = spectrum[peak_idx]
    lower_half_idx = np.nonzero(spectrum < 0.5 * peak_val)[0]
    last_idx_left = lower_half_idx[lower_half_idx < peak_idx][-1]
    firt_idx_right = lower_half_idx[lower_half_idx > peak_idx][0]

    return free_spectral_range / (xaxis[firt_idx_right] - xaxis[last_idx_left])

if __name__ == '__main__':

    phase = np.linspace(0, 2*np.pi, 1000)

    finesse = 100
    g = 2*np.pi/finesse

    signal = 0.5*g / ( (0.5*g)**2 + (phase - np.pi)**2)

    print(estimate_finesse(phase, signal))


    plt.show()