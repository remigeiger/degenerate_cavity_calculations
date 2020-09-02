import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import fftcavity1d as fc
import efield.efields as efields

import spherical_lens_zernike as zernike

""" Defining the cavity and handler """
f = 0.200
R = 0.985
TL = 0.995

cavity = fc.cavities.MIGACavity(
    f=f, d1=f, d2=f,
    R1=R, R2=R, TL=TL,
    parabolic_lens=False,
)


handler = fc.CavityHandler(cavity, radial=True)

""" Defining the input field """
lda = 852e-9
k = 2*np.pi/lda
win = 6.8e-3
x0, N = 4 * win, 2**13
x = np.linspace(-x0, x0, N)
E_ref = efields.hg_efield(
    n=0, x=x, w=win, amplitude=1.0, x0=0,
    normalize_power=True, lda=lda, radial=handler.radial,
    prop_type='evanescent',
)

n_roundtrips = int(2*cavity.finess)

factor = 0.0

aperture = 50.8e-3
aberration = factor*6.260059919214541*zernike.spherical_3(x * 2 / aperture) #- 0.0136299259746*zernike.spherical_4(x * 2 / aperture)
# aberration -= aberration[N//2]
# cavity.S1.distance -= 1e-3
# phase_interface = fc.interfaces.PhaseMask(aberration)

# cavity.insert(3, phase_interface)
phase_shifter = fc.interfaces.PhaseMask(0)
cavity.insert(-1, phase_shifter)


fig, ax_modes = plt.subplots()
fig.canvas.set_window_title('Modes')

fig, ax_spectra = plt.subplots()
fig.canvas.set_window_title('Spectra {:.1f} mm'.format(win*1e3))

phase_offset = None
for displacement in tqdm(np.linspace(-125e-6, +125e-6, 3)):
    label = '{:.1f} um'.format(displacement*1e6)
    cavity.S2.distance = f + displacement #- 780e-6*factor
    # print(cavity)
    phase_shifter.phase_map = -k * displacement

    handler.calculate_fields(E_ref, n_roundtrips)
    # fig, ax = handler.plot_fields()
    # fig.canvas.set_window_title('{:.2f}'.format(displacement*1e6))
    resonance_phases = handler.compute_resonances(n_res=1)
    if phase_offset is None:
        phase_offset = resonance_phases[0]

    phases, spectrum = handler.spectrum.spectrum(num_focused=400, focus_width=1/10)

    ax_spectra.plot((phases - phase_offset)/2/np.pi * 375, spectrum, label=label)

    mode = handler.efield_at(resonance_phases[0])
    ax_modes.plot(mode.x, mode.I / mode.I.max(), label=label)

ax_modes.plot(E_ref.x, E_ref.I / E_ref.I.max(), label='Reference')

ax_spectra.legend()
ax_modes.legend()

plt.show()

