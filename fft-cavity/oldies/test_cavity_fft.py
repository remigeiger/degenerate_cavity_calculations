from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
from math import pi, sqrt
from time import time
import scipy.optimize as so
from tqdm import tqdm

from efield import EField
from efield.efields import HG_efield
from oldies.cavitytoolbox import compute_power, compute_fields, compute_total_field
import tmp


def gaussian(x, amplitude, center, waist, offset):
    return amplitude * np.exp(-(x-center)**2/waist**2) + offset


def fermi_dirac(x, amplitude, center, radius, beta=16.25, offset=0):
    return offset + amplitude/(1 + np.exp(beta*((np.abs(x)-center)/radius - 1)))


def round_to_odd(n):
    n = int(np.ceil(n))
    return n + 1 if n % 2 == 0 else n


if __name__ == '__main__':
    """ Input Field """
    win = 2e-3
    N = 2**14
    x0 = 20*win
    x = np.linspace(-x0, x0, N)

    lda = 770e-9
    E_ref = EField(x, E=gaussian(x, amplitude=1, center=0*win,
                                 waist=win, offset=0),
                   lda=lda, normalize=True)
    E_ref = HG_efield(n=0, x=x, w=win, amplitude=1.0, x0=0.0,
                      normalize=True, lda=771e-9)
    # E_ref += HG_efield(n=1, x=x, w=win, amplitude=1.0, x0=0.0,
    #                    normalize=True, lda=771e-9)
# E_ref = EField(x, E=fermi_dirac(x, amplitude=1, center=0*win,
    #                                 radius=win, beta=16.25, offset=0),
    #                lda=lda, normalize=True)

    """ Cavity parameters """
    angle_beam = 0e-3
    lens_center = 0.0
    angle_m2 = 0e-3
    angle_m1 = 0e-3

    f = 400e-3
    d1 = f + 1000e-6
    d2 = f + 0e-6

    r1 = sqrt(0.98)
    r2 = sqrt(0.98)
    tL = sqrt(1.0)

    F = pi*sqrt(r1*r2*tL**2)/(1-r1*r2*tL**2)
    N_rt = int(2*F)

    print(F)

    # x_map, surface_map = tmp.generate_map(2*x0, lda/200, N, win/1)
    # plt.figure()
    # plt.plot(x_map/win, surface_map)
    # plt.show()

    def cavity_fun(efield):
        efield = efield.propagate(d2).lens(f, lens_center).propagate(d1)
        efield = efield.tilt(2*angle_m1)
        efield = efield.propagate(d1).lens(f, lens_center).propagate(d2)
        efield = efield.tilt(2*angle_m2)
        return efield

    def cavity_fun(efield):
        # efield = efield.transmit(np.exp(-1j*efield.k*surface_map))
        # efield = efield.lens(5)
        efield = efield.propagate(d2).lens(f).propagate(d1).propagate(d1)
        return efield.lens(f).propagate(d2)

    """ Calculation """
    input_efield = sqrt(1-r2**2)*EField.from_EField(E_ref).tilt(angle_beam)

    t = time()
    fields = compute_fields(cavity_fun, input_efield, r1*r2*tL**2, N_rt)
    t_fields = time()-t

    t = time()
    res = so.minimize_scalar(lambda dL: -compute_power(x, fields, lda, dL),
                             bounds=(0, lda/2), method='bounded',
                             options={'xatol': lda/10000})
    t_minimize = time()-t

    n = 100
    DdL = lda/2/20
    dLs = np.linspace(0, 1.2*lda/2, n)
    dLs = np.hstack([np.linspace(res.x - DdL, res.x + DdL, n),
                     np.linspace(res.x + DdL, res.x + lda/4 - DdL, 15),
                     np.linspace(res.x + lda/4 - DdL, res.x + lda/4 + DdL, n)])

    t = time()
    power = [compute_power(x, fields, lda, dL)
             for dL in tqdm(dLs, desc='Spectrum calculation')]
    t_spectrum = time()-t

    E_cav_1 = EField(x, compute_total_field(fields, lda, res.x))
    E_cav_2 = EField(x, compute_total_field(fields, lda, res.x + lda/4))


    # Timings
    print('Time for fields calculation : {:.1f} s'.format(t_fields))
    print('Time for maxima search : {:.1f} ms'.format(t_minimize*1e3))
    print('Time for {} points spectrum : {:.1f} s'.format(n, t_spectrum))

    print('Maxima found at dL = {:.3f} lda'.format(res.x/lda))

    # Plot few round-trips
    fig, (axI, axP) = E_ref.plot(label='Input', normalize=True)
    for i in range(0, N_rt, round_to_odd(N_rt/7)):
        EField(x, fields[i, :]).plot(fig,
                                     label='# {}'.format(i),
                                     normalize=False)
    axI.legend(loc='upper right')

    # Plot spectrum
    plt.figure()
    plt.semilogy(dLs/lda, power)
    plt.xlabel(r'$\Delta L$ ($\lambda$)')
    plt.semilogy([res.x/lda, res.x/lda], plt.ylim(), '--')
    plt.semilogy([res.x/lda + 0.25, res.x/lda + 0.25], plt.ylim(), '--')
    plt.ylabel('Power')

    # Plot modes
    fig, (axI, axP) = E_ref.plot(label='Input', normalize=True)
    E_cav_1.plot(fig, label='Mode 1 cavity', normalize=True)
    E_cav_2.plot(fig, label='Mode 2 cavity', linestyle='--', normalize=True)

    # from tmp2 import r as r_matlab, I as I_matlab
    # EField(r_matlab, np.sqrt(I_matlab)).plot(fig, label='Matlab calculation',
    #                                          normalize=True, linestyle=':')
    axI.set_xlim(-3*win, 3*win)
    axI.legend()

    # plt.figure()
    # plt.semilogy(E_ref.x, abs(E_ref.I/E_ref.I.max() - E_cav_1.I/E_cav_1.I.max())/(E_cav_1.I/E_cav_1.I.max()))
    # plt.semilogy(E_ref.x, abs(E_ref.I/E_ref.I.max() - E_cav_2.I/E_cav_2.I.max())/(E_cav_2.I/E_cav_2.I.max()))
    # plt.xlim(-1.5*win, 1.5*win)

    plt.show()
