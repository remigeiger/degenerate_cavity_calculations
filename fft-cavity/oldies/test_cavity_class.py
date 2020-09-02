import numpy as np
import matplotlib.pyplot as plt

import cavity.cavities as cavities
import cavity.interfaces as interfaces
from test_cavity_fft import gaussian
from cavity.cavityhandler import CavityHandler
from efield import EField
from efield.efields import HG_efield
import utils
import zernike
from timer import Timer

"""Defining the cavity and handler"""
f = 0.4

cavity = cavities.MIGACavity(f=f, d1=f+1000e-6, d2=f,
                             R1=0.98, R2=0.98, TL=1.0)
# cavity = cavities.PlanePlaneCavity(R1=0.98, R2=0.99, L=10e-2/5.0)
# cavity = cavities.TwoMirrorCavity(R1=0.97, R2=0.99, L=100e-3,
#                                   radius1=1e10, radius2=0.5)
handler = CavityHandler(cavity)

# print(cavity.waist(771e-9))
# cavity.M2.tilt(10e-6)
print(cavity[1:] + cavity[-2::-1])

"""Defining the input field"""
win = 2e-3
x0, N = 40*win, 2**14
x = np.linspace(-x0, x0, N)
E_ref = EField(x, gaussian(x, 1, 0*win, win, 0),
               lda=771e-9, normalize=True)
# E_ref = HG_efield(n=0, x=x, w=win, amplitude=1.0, x0=0*win,
#                   normalize=True, lda=771e-9)
# E_ref += HG_efield(n=0, x=x, w=win, amplitude=1.0, x0=0.0,
#                    normalize=True, lda=771e-9)
E_in = E_ref.tilt(0e-4)

r_zernike = 3e-3
u = x / r_zernike
phase_map = np.zeros(u.size)
phase_map[abs(u) < 1.0] = zernike.zernike(4, 0, 5.0, abs(u[abs(u) < 1.0]))
# cavity.insert(2, interfaces.PhaseMask(phase_map))

"""Computation"""
N_rt = int(2*cavity.finess)

with Timer() as t_fields:
    handler.calculate_fields(E_in, N=N_rt)

with Timer() as t_lock:
    phases_lock = handler.lock_cavity()


dp = 2*np.pi/20
phases = utils.focused_linspace(min(0, min(phases_lock)),
                                max(2*np.pi, max(phases_lock)),
                                phases_lock, dp,
                                num_focused=100, num_unfocused=100)
# phases = np.linspace(0, 2*np.pi, 100)

with Timer() as t_spectrum:
    spectrum = handler.spectrum(phases=phases)

modes = [handler.efield_at(phase) for phase in phases_lock]

"""Printing results"""
print('Finess : {:.0f}, {} round trips'.format(cavity.finess, N_rt))
print('Fields calculation   : {:.1f}s'.format(t_fields.interval))
print('Lock calculation     : {:.2f}s'.format(t_lock.interval))
print('Spectrum calculation : {:.1f}s'.format(t_spectrum.interval))

fig, (axI, axP) = handler.plot_fields()

fig.canvas.set_window_title('Propagated fields')
fig.suptitle('Propagated fields')
axP.set_ylim(-1, 1)
axI.legend(loc='upper right')

fig, (axI, axP) = EField.create_figure()
for i, mode in enumerate(modes[:3]):
    mode.plot(fig=fig, normalize=True, label='Mode {}'.format(i+1))

fig.canvas.set_window_title('Resonating modes')
fig.suptitle('Resonating modes')
axI.legend()
axP.set_ylim(-1, 1)


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
