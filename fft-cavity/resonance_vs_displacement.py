import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import fftcavity1d as fc
import efield.efields as efields

""" Defining the cavity and handler """
f = 0.225
R = 0.985
TL = 1.0-1e-3

cavity = fc.cavities.MIGACavity(
    f=f, d1=f, d2=f,
    R1=R, R2=R, TL=TL,
    parabolic_lens=False,
)


finess = cavity.finess
print(finess)

# cavity.S1.distance -= 1e-3
phase_shifter = fc.interfaces.PhaseMask(0)
cavity.insert(-1, phase_shifter)

handler = fc.CavityHandler(cavity, radial=True)

""" Defining the input field """
lda = 852e-9
win = 6.8e-3
x0, N = 4 * win, 2**12
x = np.linspace(-x0, x0, N)
E_ref = efields.hg_efield(
    n=0, x=x, w=win, amplitude=1.0, x0=0,
    normalize_power=True, lda=lda, radial=handler.radial,
    prop_type='evanescent',
)

# E_ref = E_ref.propagated(-200)
# E_ref.plot()

fig_modes, ax_modes = plt.subplots()
fig_modes.canvas.set_window_title('Modes')

fig_spectra, ax_spectra = plt.subplots()
fig_spectra.canvas.set_window_title('Spectra {:.1f} mm'.format(win*1e3))

phase_offset = None

for displacement in tqdm(np.linspace(-20e-6, +20e-6, 11)):
    label = '{:.1f} um'.format(displacement*1e6)
    cavity.S2.distance = f + displacement + 25e-6
    phase_shifter.phase_map = -2*np.pi/lda * displacement

    handler.calculate_fields(E_ref, int(2*finess))
    resonance_phases = handler.compute_resonances(n_res=1)
    if phase_offset is None:
        phase_offset = resonance_phases[0]

    phases, spectrum = handler.spectrum.spectrum(linspace_kwargs={'focus_width':1/15, 'num_focused':300})

    ax_spectra.plot((phases - phase_offset)/(2*np.pi)*375, spectrum, label=label)

    mode = handler.efield_at(resonance_phases[0])
    ax_modes.plot(mode.x, mode.I, label=label)

ax_spectra.legend()
ax_modes.legend()

fig_modes.tight_layout()
fig_spectra.tight_layout()

plt.show()

