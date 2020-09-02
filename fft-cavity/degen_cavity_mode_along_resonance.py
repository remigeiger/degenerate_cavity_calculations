"""Script to lock a MIGA cavity and look at the field along the resonance
"""
import logging
import numpy as np
import matplotlib.pyplot as plt

import fftcavity1d as fc
from efield.efields import hg_efield, fd_efield
import zernike

from radial_image_from_profile import image_from_profile

logging.basicConfig(level=logging.INFO)

"""Defining the cavity and handler
"""
f = 0.2
R = 0.98
Tl = 1.0
cavity = fc.cavities.MIGACavity(
    f=f, d1=f, d2=f,
    R1=R, R2=R, TL=Tl,
    parabolic_lens=False,
)
cavity.M1.tilt(0)
cavity.S2.distance += 12.5e-6
# cavity.S1.distance += +2e-3
cavity.L.tilt(0)


handler = fc.cavityhandler.CavityHandler(cavity, radial=True)

logging.info('Cavity finesse : {:.0f}'.format(int(cavity.finess)))
logging.info('Expected gain : {:.1f}'.format((1-R)/(1-Tl*R)**2))

"""Defining the input field
"""
lda = 852e-9
win = 6e-3
x0, N = 7 * win, 2**12
x = np.linspace(-x0, x0, N)
x_transverse = 0
E_ref = (
    hg_efield(
        n=0, lda=lda,
        x=x, w=win, amplitude=1.0, x0=x_transverse,
        normalize_power=True,
        radial=handler.radial,
    )
    + 0 * hg_efield(
        n=1, lda=lda,
        x=x, w=win, amplitude=1.0, x0=0,
        normalize_power=True,
    radial=handler.radial,
    )
)
# E_ref = FD_efield(
#     x=x, R0=5e-3, b=16, amplitude=1.0, x0=0,
#     normalize_power=True, lda=lda, radial=handler.radial,
# )
E_ref.E /= np.sqrt(E_ref.P)

# E_ref.propagate(-100)
# logging.info('k s0 = {:.2f} '.format(2*np.pi/lda * lda*f/(np.pi * win) / np.sqrt(2)))

"""Adding aberrations
"""
r_optic = 2.5e-2

orders = [(4, 0)]
coeffs = [2 * np.pi * 0.02]
phase_map = zernike.zernike_combination(orders, np.array(coeffs),
                                        x / r_optic, phi=0.0)

# phase_map = generate_map(N, 0.04) * 2 * np.pi * 0.1
# cavity.insert(2, PhaseMask(-phase_map))

# plt.figure()
# plt.plot(x / win, phase_map / (2 * np.pi))

"""Computation
"""
N_rt = int(2 * cavity.finess)

handler.calculate_fields(E_ref, N=N_rt)
handler.plot_fields()
phases_lock = handler.compute_resonances(n_res=1)
phases, spectrum = handler.spectrum()

best_phase = phases_lock[0]

logging.info('Best gain : {:.1f}'.format(handler.power(best_phase)))

res_scan_width = np.pi / 10
scan_lims = (best_phase - res_scan_width / 2,
             best_phase + res_scan_width / 2)
n_res_scan = 5**2

modes = [handler.efield_at(phase)
         for phase
         in np.linspace(scan_lims[0], scan_lims[1],
                        n_res_scan)]

best_mode = handler.efield_at(best_phase)

"""Printing results
"""
def stack_images(images, n_per_row=5):
    lines = []
    for p in range(len(images) // n_per_row):
        images_in_row = images[p * n_per_row:(p + 1) * n_per_row]
        if len(images_in_row) < n_per_row:
            images_in_row += [np.zeros(images_in_row[0].shape)
                              for k in range(n_per_row - len(images_in_row))]
        lines.append(np.concatenate(images_in_row, axis=1))
    return np.concatenate(lines)

images = [image_from_profile(mode.x, mode.I, N=150, scale=0.2)
          for mode in modes]
bigimg = stack_images(images, n_per_row=5)

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

for lim in scan_lims:
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

# fig, ax = plt.subplots()
# for num, mode in enumerate(modes):
#     plt.plot(mode.x, mode.I, label=num)
# plt.legend()

plt.show()

