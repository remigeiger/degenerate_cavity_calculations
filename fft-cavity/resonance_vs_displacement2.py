import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import fftcavity1d as fc
import efield.efields as efields

""" Defining the cavity and handler """
f = 0.200
R = 0.985
TL = 1.0-1e-3

cavity = fc.cavities.MIGACavity(
    f=f, d1=f, d2=f,
    R1=R, R2=R, TL=TL,
    parabolic_lens=False,
)
# cavity.L.f *= 0.999
phase_shifter = fc.interfaces.PhaseMask(0)
cavity.insert(-1, phase_shifter)
handler = fc.CavityHandler(cavity, radial=True)

finess = cavity.finess

def do_the_thing(waist, d2_ref):

    """ Defining the input field """
    lda = 852e-9
    win = waist
    x0, N = 4 * win, 2**13
    x = np.linspace(-x0, x0, N)
    E_ref = efields.hg_efield(
        n=0, x=x, w=win, amplitude=1.0, x0=0,
        normalize_power=True, lda=lda, radial=handler.radial,
        # prop_type='evanescent',
        prop_type='evanescent',
    )


    fig_modes, (ax_modes_1, ax_modes_2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    fig_modes.canvas.set_window_title('Modes {:.1f} mm waist'.format(waist*1e3))
    ax_modes_1.set_title('Modes with {:.1f} mm input waist'.format(waist*1e3))
    ax_modes_1.set_ylabel('Intensity (au)')
    ax_modes_2.set_ylabel('Intensity (au)')
    ax_modes_2.set_xlabel('x (mm)')

    fig_spectra, (ax_spectra_1, ax_spectra_2) = plt.subplots(2, 1, sharex=True, figsize=(6, 6))
    fig_spectra.canvas.set_window_title('Spectra {:.1f} mm waist'.format(waist*1e3))
    ax_spectra_1.set_title('Resonances with {:.1f} mm input waist'.format(waist*1e3))
    ax_spectra_1.set_ylabel('Optical Gain')
    ax_spectra_2.set_ylabel('Optical Gain')
    ax_spectra_2.set_xlabel('Frequency (MHz)')

    center_phase = None
    for displacement in np.linspace(0, 6*25e-6, 7):
        label = '{:.1f} um'.format(displacement*1e6)
        cavity.S2.distance = f + displacement + d2_ref
        phase_shifter.phase_map = -2*np.pi/lda * displacement

        handler.calculate_fields(E_ref, int(2*finess))
        resonance_phases = handler.compute_resonances(n_res=1)
        if center_phase is None:
            center_phase = resonance_phases[0]

        phases, spectrum = handler.spectrum.spectrum(focus_width=1/5, num_focused=200)

        ax_spectra_1.plot((phases - center_phase)/(2*np.pi)*375, spectrum, label=label)

        mode = handler.efield_at(resonance_phases[0])
        ax_modes_1.plot(mode.x*1e3, mode.I, label=label)

        # Other direction
        label = '-{:.1f} um'.format(displacement*1e6)
        cavity.S2.distance = f - displacement + d2_ref
        phase_shifter.phase_map = +2*np.pi/lda * displacement

        handler.calculate_fields(E_ref, int(2*finess))
        resonance_phases = handler.compute_resonances(n_res=1)


        phases, spectrum = handler.spectrum.spectrum(focus_width=1/4, num_focused=200)

        ax_spectra_2.plot((phases - center_phase)/(2*np.pi)*375, spectrum, label=label)

        mode = handler.efield_at(resonance_phases[0])
        ax_modes_2.plot(mode.x * 1e3, mode.I, label=label)



    ax_spectra_1.legend()
    ax_spectra_2.legend()
    ax_modes_1.legend()
    ax_modes_2.legend()

    fig_modes.tight_layout()
    fig_spectra.tight_layout()

    # fig_modes.savefig('images/modes_{:.1f}mm_waist.png'.format(waist*1e3))
    # fig_spectra.savefig('images/spectra_{:.1f}mm_waist.png'.format(waist*1e3))

cfgs = [
    # {'waist': 0.7e-3, 'd2_ref': 0},
    # {'waist': 1.6e-3, 'd2_ref': 8.3e-6},
    # {'waist': 3.4e-3, 'd2_ref': 16.7e-6},
    # {'waist': 5.1e-3, 'd2_ref': 25e-6},
    {'waist': 6.8e-3, 'd2_ref': 0*25e-6},
    # {'waist': 7e-3, 'd2_ref': 30e-6},
    # {'waist': 5.1e-3, 'd2_ref': 0}
]

for cfg in tqdm(cfgs):
    do_the_thing(**cfg)

plt.show()
