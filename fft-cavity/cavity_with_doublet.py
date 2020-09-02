import numpy as np
import matplotlib.pyplot as plt
import logging

import fftcavity1d as fc
import fftcavity1d.cavities as cavities
from fftcavity1d.cavityhandler import CavityHandler
from fftcavity1d.interfaces import PhaseMask
from efield import EField
from efield.efields import hg_efield, fd_efield

import zernike
from generate_map import generate_map
from autofit_gaussian import autofit_gaussian, gaussian2


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" Defining the cavity and handler """
f = 0.2
R = 0.98
TL = 1.0

cavity = fc.cavity.Cavity()

cavity.append(fc.interfaces.FlatMirror(R))
cavity.append(fc.interfaces.FreeSpace(408.455687e-3 + 10e-3 - 6.89e-3))
cavity.append(fc.interfaces.SphericalLens(f=114.444e-3, T=TL))
cavity.append(fc.interfaces.FreeSpace(36.73e-3 + 6.89e-3))
cavity.append(fc.interfaces.SphericalLens(f=-114.444e-3, T=TL))
# cavity.append(fc.interfaces.FreeSpace(183.81671435e-3 + 2e-3))
cavity.append(fc.interfaces.FreeSpace(183.84e-3 + 2e-3))
cavity.append(fc.interfaces.FlatMirror(R))


handler = CavityHandler(cavity, radial=False)
logging.info('Expected gain : {:.1f}'.format((1-R)/(1-TL*R)**2))

""" Defining the input field """
lda = 852e-9
win = 2e-3
x0, N = 10 * win, 2**15
x = np.linspace(-x0, x0, N)
x_trans = 0
E_ref = (
        1 * hg_efield(
                n=0, x=x, w=win, amplitude=1.0, x0=x_trans,
                normalize_power=True, lda=lda, radial=handler.radial
        ) +
        0 * hg_efield(
                n=1, x=x, w=win, amplitude=1.0, x0=0,
                normalize_power=True, lda=lda, radial=handler.radial
        )
)
# E_ref = fd_efield(x=x, R0=5e-3, b=16, amplitude=1.0, x0=0,
#         normalize_power=True, lda=lda, radial=handler.radial)
E_ref.radial = handler.radial
E_ref.E /= np.sqrt(E_ref.P)
# E_ref = E_ref.tilt(0).propagate(20)

fig, (axI, axP) = EField.create_figure()
E_ref.plot(fig=fig, normalize=True, label='Initial')
# fields = cavity.propagate_efield(E_ref, N=100) #.plot(fig=fig, label='One round trip')

# E_out = EField.from_EField(E_ref)
# E_out.E = fields[-1, :]
# E_out.plot(fig=fig, normalize=True, label='Out')
#
# axP.set_ylim(-0.05, 0.15)
#
# axI.legend()
#
# plt.show()
# exit()

""" Computation """
r_optic = 25e-3
orders = [(0, 0), (1, -1), (1, 1), (2, 0),
          (2, -2), (2, 2), (3, -1), (3, 1), (4, 0)]
coeffs = [0, 0, 0, 0,
          0.012, -0.011, 0.072, 0.040, 0.109]

orders = [(4, 0)]
coeffs = [0.5]
# phase_map = zernike.zernike_combination(orders, np.array(coeffs),
#                                         x / r_optic, phi=0.0)

# plt.figure()
# plt.plot(x/win, phase_map/(2*np.pi))
# cavity.insert(1, PhaseMask(phase_map))

# phase_map = generate_map(len(x), 0.2) * 0.025 * 2 * np.pi
# cavity.insert(1, PhaseMask(phase_map))
# cavity.insert(3, PhaseMask(phase_map))

logger.info('Cavity Finesse : {}'.format(int(cavity.finess)))
N_rt = int(3 * cavity.finess)
handler.calculate_fields(E_ref, N=N_rt)
resonance_phases = handler.compute_resonances(n_res=2)
modes = [handler.efield_at(phase) for phase in resonance_phases]

beam_right = modes[0].propagated(3e-2)
beam_left = modes[0].propagated(-3e-2)

logger.info('Best gain : {}'.format(beam_right.P))

""" Printing results """
# Fields during the propagation
fig, (axI, axP) = handler.plot_fields()

fig.canvas.set_window_title('Propagated fields')
fig.suptitle('Propagated fields')
# axP.set_ylim(-1.2, 0.1)
# axP.set_xlim(-4, 4)
axI.legend(loc='upper right')


# Resonating modes
fig, (axI, axP) = EField.create_figure()
for i, mode in enumerate(modes):
    mode.plot(fig=fig, normalize=False, label='Mode {}'.format(i + 1))
    if i == 0:
        popt, fit = autofit_gaussian(mode.x, mode.I)

axI.plot(mode.x * 1e3, gaussian2(mode.x, *popt),
         linestyle='--', color=[0.2] * 3, linewidth=1,
         label='Mode 1 fit')

fig.canvas.set_window_title('Resonating modes {0:.0f}'.format(fit['waist'] * 1e6))
fig.suptitle('Resonating modes')
axI.legend()
axP.set_ylim(-10, 1)

# logger.info('Fitted waist : {0:.0f} um'.format(fit['waist'] * 1e6))

# Spectrum of the cavity
fig, ax = plt.subplots()
handler.plot_spectrum(ax=ax, show_resonances=True)

fig.canvas.set_window_title('Cavity spectrum')
fig.suptitle('Cavity spectrum')
ax.set_xlabel(r'Phase ($\pi$)')
ax.set_ylabel('Gain')
ax.grid(which='both')
ax.grid(which='minor', color=[0.9] * 3)
ax.legend()

# Propagated mode
idx = (-win*20 < x) & (x < win*20)
relative_phase = np.unwrap(beam_right.phase - beam_left.phase)
fig, (axI, axP) = EField.create_figure()
# axI.plot(x[idx], beam_left.I[idx]/beam_left.I.max())
axI.plot(x[idx]*1e3, beam_right.I[idx]/beam_right.I.max())
x_plot = x[idx]
phase_plot = (relative_phase[idx] - relative_phase[N//2])*1e3

lines = axP.plot(x_plot[abs(x_plot) < win]*1e3, phase_plot[abs(x_plot) < win])
axP.plot(x_plot[x_plot > win]*1e3, phase_plot[x_plot > win],
         color=lines[0].get_color(), linestyle='--')
axP.plot(x_plot[x_plot < win]*1e3, phase_plot[x_plot < win],
         color=lines[0].get_color(), linestyle='--')
# axI.set_xlim(-win, win)

axI.set_ylabel('Intensity (au)')
axP.set_ylabel('Relative laser phase (mrad)')
axP.set_ylim(-2, 2)

plt.show()
