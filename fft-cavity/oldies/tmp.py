import numpy as np
import matplotlib.pyplot as plt

from efield.efields import hg_efield
from fftcavity1d.interfaces import ParabolicLens, SphericalLens

lens = ParabolicLens(f=20e-2)

w = 5e-3
num = 2**14
x = np.linspace(-7*w, 7*w, num)

E = hg_efield(0, x, w, prop_type='fresnel')

E_1 = lens(E.propagated(lens.f)).propagated(lens.f)
E_out = lens(E_1.propagated(lens.f)).propagated(lens.f)

fig, (axI, axP) = E.plot(normalize=True, label='in',)
E_1.plot(fig=fig, normalize=True, label='1')
E_out.plot(fig=fig, normalize=True, label='out')

rel_phase = np.unwrap(E.phase - E_out.phase)

_, ax = plt.subplots()
# ax.plot(x, 1 - abs(E_out.E/E.E))
ax.plot(x, rel_phase - rel_phase[num//2])

ax2 = ax.twinx()
ax2.fill(x, E.I, 'k', alpha=0.1)

ax.set_xlim([-3*w, 3*w])
ax.set_ylim([-1e-1, 1e-1])

print(E.lda * lens.f / np.pi / w * 1e3)

plt.show()