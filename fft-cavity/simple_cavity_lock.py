import numpy as np
import matplotlib.pyplot as plt
# import scipy.integrate as si
import time

import efield
import fftcavity1d as fc


win = 6.8e-3
lda = 852e-9
f = 0.2
radial = False
xm = 7 * win
nx = 2**14
x = np.linspace(-xm, xm, nx)

E_in = efield.efields.hg_efield(n=0, x=x, w=win, normalize_power=True,
                                x0=0, lda=lda, radial=radial)

cavity = fc.cavities.MIGACavity(f=f, d1=f, d2=f,
                                R1=0.98, R2=0.98, TL=1.0-1e-3,
                                parabolic_lens=False)
# cavity.S1.distance += 250e-6
# cavity.S2.distance += 500e-6

handler = fc.CavityHandler(cavity, radial=radial)

n_roundtrips = int(2 * cavity.finess)

t = time.time()
handler.calculate_fields(E_in, N=n_roundtrips)
print(time.time() - t)
resonance_phases = handler.compute_resonances(n_res=4)

print('Maximum gain : {0:.1f}'.format(handler.power(resonance_phases[0])))
print('Resonances at :')
print(resonance_phases)

fig, ax = plt.subplots()
handler.plot_spectrum(ax=ax, show_resonances=True)
ax.grid(which='both')
ax.grid(which='minor', color=[0.9] * 3)
ax.set_xlabel('Phase (rad)')
ax.set_ylabel('Optical Gain')

fig, _ = efield.EField.create_figure()
for p in resonance_phases:
    handler.efield_at(p).plot(fig=fig, normalize=True)

# efields.HG_efield(n=6, x=x, w=win, normalize_power=True,
#                   x0=0, lda=lda, radial=radial).plot(fig=fig, normalize=True)

E_in.plot(fig=fig, normalize=True, ls=':')

plt.show()
