import logging
import numpy as np
import matplotlib.pyplot as plt

import fftcavity1d as fc
import efield


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%m-%d %H:%M',
                    filename='test.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
console.setFormatter(formatter)
logging.getLogger().addHandler(console)


"""Parameters
"""
f = 0.2
R = 0.95
Tl = 1.0

lda = 852e-9
win = 6e-3
N = 2**13

"""Defining the cavity
"""
cavity = fc.cavities.MIGACavity(f=f, d1=f, d2=f,
                                R1=R, R2=R, TL=Tl)
handler = fc.cavityhandler.CavityHandler(cavity=cavity)

"""Defining the input field
"""
x = np.linspace(-10*win, 10*win, N)
input_efield = efield.efields.HG_efield(n=0, x=x, w=win, lda=lda,
                        radial=handler.radial, normalize_power=True)

"""Calculation
"""
fig_spectrum, ax_spectrum = plt.subplots()
delta2s = np.linspace(-200e-6, 200e-6, 5)
for delta2 in delta2s:
    logging.info('delta : {:.0f} um'.format(delta2 * 1e6))
    cavity.S2.distance = f + delta2
    handler.calculate_fields(input_efield, N=3*int(cavity.finess))

    handler.compute_resonances()
    phases, spectrum = handler.spectrum()
    phase_of_max = phases[np.argmax(spectrum)]

    ax_spectrum.semilogy(phases - phase_of_max, spectrum,
                         label=r'$\delta$={: 4.0f} um G={:5.1f}'.format(delta2 * 1e6,
                                                                   max(spectrum)))

ax_spectrum.legend()
ax_spectrum.grid()
ax_spectrum.set_xlabel('Phase (rad)')
ax_spectrum.set_ylabel('Optical gain')

"""Info
"""
logging.info('Small waist size : {:.2f} um'.format(lda * f / np.pi / win * 1e6))

plt.show()
