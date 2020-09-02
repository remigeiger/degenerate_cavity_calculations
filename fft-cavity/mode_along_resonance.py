"""Script to lock a MIGA cavity and look at the field along the resonance
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

import fftcavity1d as fc
from efield.efields import hg_efield

from radial_image_from_profile import image_from_profile

logging.basicConfig(level=logging.INFO)

"""Defining the cavity and handler
"""
f = 0.2
R = 0.985
Tl = 0.99
cavity = fc.cavities.MIGACavity(
    f=f, d1=f, d2=f,
    R1=R, R2=R, TL=Tl,
    parabolic_lens=False,
)
cavity.M1.tilt(0)
cavity.S2.distance -= 40e-6
cavity.S1.distance -= 50e-6
cavity.L.tilt(0)


handler = fc.cavityhandler.CavityHandler(cavity, radial=True)

logging.info('Cavity finesse : {:.0f}'.format(int(cavity.finess)))
logging.info('Expected gain : {:.1f}'.format((1-R)/(1-Tl*R)**2))

"""Defining the input field
"""
lda = 852e-9
win = 6e-3
x0, N = 5 * win, 2**13
x = np.linspace(-x0, x0, N)
x_transverse = 0
E_ref = (
    hg_efield(
        n=0, lda=lda,
        x=x, w=win, amplitude=1.0, x0=x_transverse,
        normalize_power=True,
        radial=handler.radial,
    )
)


"""Computation
"""
N_rt = int(2 * cavity.finess)

handler.calculate_fields(E_ref, N=N_rt)
handler.plot_fields()
phases_lock = handler.compute_resonances(n_res=1)
phases, spectrum = handler.spectrum()

best_phase = phases_lock[0]

logging.info('Best gain : {:.1f}'.format(handler.power(best_phase)))

res_scan_width = np.pi / 50
scan_lims = (best_phase - res_scan_width,
             best_phase + res_scan_width)
n_res_scan = 3

side_phases = [best_phase - np.pi/40, best_phase + np.pi/60]

# modes = [handler.efield_at(phase)
#          for phase
#          in np.linspace(scan_lims[0], scan_lims[1],
#                         n_res_scan)]

modes = [
    handler.efield_at(phase)
    for phase in [side_phases[0], best_phase, side_phases[1]]
]

best_mode = handler.efield_at(best_phase)

"""Printing results
"""
def stack_images(images, n_per_row=5):
    lines = []
    for p in range(len(images) // n_per_row):
        images_in_row = images[p * n_per_row:(p + 1) * n_per_row]
        if len(images_in_row) < n_per_row:
            images_in_row += [np.zeros(images_in_row[0].shape)
                              for _ in range(n_per_row - len(images_in_row))]
        lines.append(np.concatenate(images_in_row, axis=1))
    if len(lines) == 1:
        return lines[0]
    return np.concatenate(lines)

images = [image_from_profile(mode.x, mode.I, N=150, scale=0.2)
          for mode in modes]
bigimg = stack_images(images, n_per_row=n_res_scan)

fig_img, ax_img = plt.subplots()
ax_img.imshow(bigimg.astype(int), cmap='gray')


fig, ax = plt.subplots()
ylim = [min(spectrum), max(spectrum)]
for i, phase in enumerate(phases_lock):
    ax.plot([phase / np.pi] * 2, ylim,
            linestyle='--', linewidth=0.8,
            label='Mode {}'.format(i + 1))

ax.plot(phases / np.pi, spectrum,
            label='Cavity spectrum', color=[0.4] * 3)

for lim in side_phases:
    ax.plot([lim / np.pi] * 2, ylim,
                linestyle='--', color=[0.3] * 3)

fig.canvas.set_window_title('Cavity spectrum')
fig.suptitle('Cavity spectrum')
ax.set_xlabel(r'Phase ($\pi$)')
ax.set_ylabel('Gain')
ax.grid(which='both')
ax.grid(which='minor', color=[0.9] * 3)
ax.legend()


fig, ax = best_mode.plot()

plt.show()

