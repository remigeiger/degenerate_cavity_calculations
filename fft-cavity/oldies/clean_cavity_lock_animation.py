import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as colors

import fftcavity1d.cavities as cavities
from fftcavity1d.cavityhandler import CavityHandler
from fftcavity1d.interfaces import PhaseMask
from efield import EField
from efield.efields import HG_efield, FD_efield
# import utils
# import zernike

from radial_image_from_profile import image_from_profile


"""Defining the cavity and handler"""
f = 0.2
cavity = cavities.MIGACavity(f=f, d1=f, d2=f,
                             R1=0.985, R2=0.985, TL=1.0 - 1e-3)
# cavity = cavities.PlanePlaneCavity(R1=0.98, R2=0.98, L=f)
handler = CavityHandler(cavity)
handler.radial = True
cavity.M1.tilt(0)
# cavity.S2.distance += +30e-6
cavity.L.tilt(0)

"""Defining the input field"""
lda = 852e-9
win = 3e-3
x0, N = 10 * win, 2**15
x = np.linspace(-x0, x0, N)
E_ref = (HG_efield(n=0, x=x, w=win, amplitude=1.0, x0=0,
                   normalize_power=True, lda=lda)
         + 0 * HG_efield(n=1, x=x, w=win, amplitude=1.0, x0=0,
                         normalize_power=True, lda=lda))
# E_ref = FD_efield(x=x, R0=5e-3, b=16, amplitude=1.0, x0=0.0, lda=lda, normalize_power=True)
E_ref.radial = handler.radial
E_ref.E /= np.sqrt(E_ref.P)
# E_ref = E_ref.tilt(1e-6).propagate(0)

"""Computation"""
r_optic = 2.5e-2
orders = [(0, 0), (1, -1), (1, 1), (2, 0),
          (2, -2), (2, 2), (3, -1), (3, 1), (4, 0)]
coeffs = [0, 0, 0, 0,
          0.012, -0.011, 4 * 0.072, 0.040, 0 * 0.109]

# orders = [(4, 0)]
# coeffs = [2 * np.pi * 0.0509]
# phase_map = zernike.zernike_combination(orders, np.array(coeffs),
#                                         x / r_optic, phi=0.0)
# mask = np.cos(2 * np.pi * x / r_optic) * ((-r_optic < x) & (x < r_optic))

# plt.figure()
# plt.plot(x / win, phase_map / (2 * np.pi))
# cavity.insert(2, PhaseMask(-phase_map))

print(int(cavity.finess))
N_rt = int(2 * cavity.finess)
# N_rt = 30

handler.calculate_fields(E_ref, N=N_rt)
# shape = handler.fields.shape

phases_lock = handler.compute_resonances(n_res=3)

# phases = utils.focused_linspace(min(0, min(phases_lock)),
#                                 max(2 * np.pi, max(phases_lock)),
#                                 phases_lock, 2 * np.pi / 20,
#                                 num_focused=100, num_unfocused=100)
phases, spectrum = handler.spectrum() # phases=phases)


best_phase = sorted(phases_lock, key=lambda p: handler.power(p))[0]
print(best_phase / np.pi)

lim = np.pi / 50
modes = [handler.efield_at(phase)
         for phase in np.linspace(-lim, lim, 15) + best_phase]

beam_right = modes[0].propagate(3e-2)
beam_left = modes[0].propagate(-3e-2)

"""Printing results"""
# fig, (axI, axP) = handler.plot_fields()
#
# fig.canvas.set_window_title('Propagated fields')
# fig.suptitle('Propagated fields')
# axP.set_ylim(-1.2, 0.1)
# axP.set_xlim(-4, 4)
# axI.legend(loc='upper right')

# fig, (axI, axP) = EField.create_figure()
images = []
max_images = 0
for i, mode in enumerate(modes):
    # mode.plot(fig=fig, normalize=False, label='Mode {}'.format(i+1))
    images.append(image_from_profile(mode.x, mode.I, N=150))
    if max(images[-1].flatten()) > max_images:
        max_images = max(images[-1].flatten())

def stack_images(images, n_per_row=5):
    lines = []
    for p in range(len(images) // n_per_row + 1):
        images_in_row = images[p * n_per_row:(p + 1) * n_per_row]
        if len(images_in_row) < n_per_row:
            images_in_row += [np.zeros(images_in_row[0].shape)
                              for k in range(n_per_row - len(images_in_row))]
        lines.append(np.concatenate(images_in_row, axis=1))
    return np.concatenate(lines)


# for image in images:
#     _, ax = plt.subplots()
#     scaled = image/max_images * 255
#     ax.imshow(scaled.astype(int))

images = [image/max_images*255 for image in images]

bigimg = stack_images(images, n_per_row=4)
fig_img, ax_img = plt.subplots()
ax_img.imshow(bigimg.astype(int), cmap='gray')
# for num, image in enumerate(images):
    # fig_img, ax_img = plt.subplots()
    # fig_img.canvas.set_window_title(f'{num}')
    # ax_img.imshow(image.astype(int),
    #               norm=colors.Normalize(vmin=0, vmax=255),
    #               interpolation='bicubic',
    #               cmap='gray')


fig_anim, ax_anim = plt.subplots()
ims = ax_anim.imshow(images[0])
def animate(img):
    ims = ax_anim.imshow(img.astype(int),
                         norm=colors.Normalize(vmin=0, vmax=255),
                         interpolation='bicubic',
                         cmap='gray')
    return ims,
# anim = animation.FuncAnimation(fig_anim, animate, images,
#                                interval=50, blit=True)

# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=30, bitrate=1800)
# anim.save('test.mp4', writer='ffmpeg')
# print('Animation saved')

fig, ax = plt.subplots()
ylim = [min(spectrum), max(spectrum)]
for i, phase in enumerate(phases_lock):
    ax.plot([phase/np.pi]*2, ylim, linestyle='--', linewidth=0.8,
            label='Mode {}'.format(i+1))
ax.semilogy(phases/np.pi, spectrum, label='Cavity spectrum', color=[0.4]*3)

fig.canvas.set_window_title('Cavity spectrum')
fig.suptitle('Cavity spectrum')
ax.set_xlabel(r'Phase ($\pi$)')
ax.set_ylabel('Gain')
ax.grid(which='both')
ax.grid(which='minor', color=[0.9]*3)
ax.legend()

plt.show()
