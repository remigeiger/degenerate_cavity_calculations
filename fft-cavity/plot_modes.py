import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import fftcavity1d as fc
import efield.efields as efields

import spherical_lens_zernike as zernike

""" Defining the cavity and handler """
f = 0.200
R = 0.985
TL = 1.0-1e-3

cavity = fc.cavities.MIGACavity(
    f=f, d1=f, d2=f,
    R1=R, R2=R, TL=TL,
    parabolic_lens=True,
)


handler = fc.CavityHandler(cavity, radial=True)

""" Defining the input field """
lda = 852e-9
k = 2*np.pi/lda
win = 5.1e-3
x0, N = 4 * win, 2**13
x = np.linspace(-x0, x0, N)
E_ref = efields.hg_efield(
    n=0, x=x, w=win, amplitude=1.0, x0=0,
    normalize_power=True, lda=lda, radial=handler.radial,
    prop_type='evanescent',
)

displacement = 42e-6

factor = 1.0

aperture = 50.8e-3
aberration = factor*6.260059919214541*zernike.spherical_3(x * 2 / aperture) #- 0.0136299259746*zernike.spherical_4(x * 2 / aperture)
# cavity.S1.distance -= 1e-3
phase_interface = fc.interfaces.PhaseMask(aberration)

cavity.insert(3, phase_interface)

cavity.S2.distance = f + displacement - 780e-6 * factor

handler.calculate_fields(E_ref, N=250)
resonance_phases = handler.compute_resonances(n_res=2)

phases, spectrum = handler.spectrum.spectrum(num_focused=400, focus_width=1/10)

fig, ax = plt.subplots()
ax.plot(phases, spectrum, color='k')
ylim = ax.get_ylim()

phases_of_interest = [3.5144, 3.54, 3.559, 3.637]

for phase in phases_of_interest:
    ax.plot([phase]*2, ylim)

modes = [handler.efield_at(phase) for phase in phases_of_interest]
fig, ax = plt.subplots()
for mode in modes:
    ax.plot(mode.x*1e3, mode.I)





plt.show()

