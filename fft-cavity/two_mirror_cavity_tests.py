""" Two mirror cavity tests
"""

import logging

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as so

import fftcavity1d as fc
import efield

from generate_map import generate_map


logging.basicConfig(level=logging.INFO)

def main():
    lda = 650e-9

    radius1 = 1e50
    radius2 = 0.4
    L = 0.2
    R = 0.98

    cavity = fc.cavities.TwoMirrorCavity(R1=R, R2=R, L=L, radius1=radius1, radius2=radius2)

    try:
        w_cav = cavity.waist(wavelength=lda)
        z_w = cavity.waist_position(wavelength=lda)
    except ZeroDivisionError:
        logging.info('Degenerate cavity, using arbitrary waist')
        w_cav = 1e-3 # 550e-6
        z_w = 0
    w_cav = 1e-3
    z_w = 0

    win = w_cav
    N = 2**13
    radial = False
    xm = 10*win
    x = np.linspace(-xm, xm, N)

    input_efield = efield.efields.HG_efield(n=0, x=x, w=win, normalize_power=True, radial=radial, lda=lda)
    # input_efield = efield.efields.FD_efield(x=x, R0=w_cav/2, b=16.25, normalize_power=True, radial=radial, lda=lda)
    # input_efield += efield.efields.HG_efield(n=1, x=x, w=win, normalize_power=True, radial=radial, lda=lda)
    input_efield = input_efield.propagate(-z_w)

    c_l = 0.5e-3

    logging.info('Expected gain : {:.0f}'.format(1/(1-R)))
    logging.info('Cavity waist : {:.10e} m'.format(w_cav))
    logging.info('Cavity waist position : {:.10e} m'.format(z_w))

    handler = fc.cavityhandler.CavityHandler(cavity, radial=radial)

    phase_map = generate_map(N=N, c_l = c_l / 2 / xm)

    # plt.figure()
    # plt.plot(x*1e3, phase_map)

    # cavity.insert(1, fc.interfaces.PhaseMask(phase_map * 2 * np.pi / 4))

    handler.calculate_fields(input_efield, N=int(2*cavity.finess))
    res_phases = handler.compute_resonances(n_init=500, n_res=5)
    best_mode = handler.efield_at(res_phases[0]).propagate(z_w)

    logging.info('Best mode gain : {:.4f}'.format(best_mode.P))

    handler.plot_spectrum(show_resonances=True)
    handler.plot_fields()

    fig, (axI, _) = efield.EField.create_figure()
    best_mode.plot(fig=fig, normalize=True, label='Highest mode')
    for num, phase in enumerate(res_phases[1:4]):
        # handler.efield_at(phase).plot(fig=fig, normalize=True, label='mode {}'.format(num+1))
        handler.efield_at(phase).propagate(z_w).plot(fig=fig, normalize=True, label='mode {}'.format(num+1))
    input_efield.plot(fig=fig, normalize=True, label='Input field', color='k')

    def squared_gaussian(xdata, waist, amplitude, center):
        """Squared gaussian enveloppe
        """
        return amplitude * np.exp(-2 * (xdata - center)**2 / waist**2)

    guess = [win, best_mode.I.max(), 0]
    popt, _ = so.curve_fit(squared_gaussian, best_mode.x, best_mode.I, p0=guess)

    fitted_efield = efield.EField.from_EField(best_mode)
    fitted_efield.E = np.sqrt(squared_gaussian(x, *popt))

    fitted_efield.plot(fig=fig, normalize=True, label='Fitted gaussian', linestyle='--')

    axI.legend()

    logging.info('Fitted gaussian waist : {:.10e} m'.format(popt[0]))

    plt.show()

if __name__ == '__main__':
    main()