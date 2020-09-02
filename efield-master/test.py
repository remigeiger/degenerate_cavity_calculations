import numpy as np
import matplotlib.pyplot as plt

from efield import EField
from efields import HG_efield


def gaussian(x, amplitude, center, waist, offset):
    return amplitude * np.exp(-(x-center)**2/waist**2) + offset


N = 2**15
win = 0.5e-3
x0 = 50*win
x = np.linspace(-x0, x0, N)

E = HG_efield(n=0, x=x, w=win,)
E.E = E.E / np.sqrt(E.I.max())

angle = 1e-2

print('Electric Field E : {}'.format(E))
print('Power : {:.5f}'.format(E.P))
print(len(E))

dist = 2.0
E_prop = E.tilt(angle).propagate(dist=dist)

zr = np.pi * win**2 / E.lda
w_th = win*np.sqrt(1+(dist/zr)**2)
E_theo = EField(x, gaussian(x, amplitude=np.sqrt(win/w_th),
                            center=angle * dist, waist=w_th, offset=0))
E_theo = HG_efield(n=0, x=x, amplitude=np.sqrt(win/w_th),
                   w=w_th, x0=angle*dist)


# plt.figure()
# plt.plot(x, abs(E_theo.I/E_theo.I.max()-E_prop.I/E_prop.I.max()))

s = 'Result agrees with theory : {}'
print(s.format(np.allclose(E_theo.I,  E_prop.I)))

fig, (axI, _) = E.plot(label='Reference')
E_prop.plot(fig, label='Propagated')
E_theo.plot(fig, linestyle='--', label='Theoretic')
axI.legend()
plt.show()