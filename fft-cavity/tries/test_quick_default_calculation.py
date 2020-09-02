import numpy as np
import matplotlib.pyplot as plt

import zernike
from efield import EField, efields
from fftcavity1d.spectrum import Spectrum

if __name__ == '__main__':
    win = 6e-3
    lda = 852e-9
    x0 = 3 * win
    N = 2**10
    x = np.linspace(-x0, x0, N)
    E = efields.HG_efield(n=0, x=x, w=win, amplitude=1.0, x0=0.0,
                          normalize=True, lda=lda)

    r_optic = 2.5e-2
    orders, coeffs = [(4, 0)], [0.5]
    phase_map = zernike.zernike_combination(orders, coeffs,
                                            x / r_optic, phi=0.0)

    R1, R2 = 0.98, 0.98
    T = 1 - R1

    base = np.sqrt(R1 * R2) * np.exp(1j * phase_map)

    def power(phase):
        u = x[len(x) // 2:]
        v = E.E[len(x) // 2:]
        b = base[len(x) // 2:]
        return np.trapz(x=u,
                        y=(2 * np.pi * u * abs(v * np.sqrt(T) / (1 - b * np.exp(1j * phase)))**2))

    S = Spectrum(power_fun=power)
    S.compute_resonances()
    phases, spectrum = S.spectrum()

    fig, ax = plt.subplots()
    ax.semilogy(phases, spectrum)
    ax.grid(which='both')
    ax.grid(which='minor', color=[0.9] * 3, linewidth=0.5)

    modes = [E.transmit(((np.sqrt(T) / (1 - base * np.exp(1j * phase)))))
             for phase in S.resonance_phases]

    fig, (axI, axP) = EField.create_figure()
    for mode in modes:
        print(mode.P)
        mode.plot(fig, normalize=False)

    plt.show()
