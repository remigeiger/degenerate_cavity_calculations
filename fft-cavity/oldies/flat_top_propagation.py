import numpy as np
from EField import EField
import matplotlib.pyplot as plt
import seaborn
import scipy.optimize as so

seaborn.set_style('white')


def fermi_dirac(xdata, amplitude, center, radius):
    return amplitude/(1 + np.exp(16.8*(abs(xdata-center)/radius-1)))


R = 1.5e-3
x0 = 25*R
N = 2**12
x = np.linspace(-x0, x0, N)

E = EField(x, np.sqrt(fermi_dirac(x, 1, 0, R)))

fig, (axI, axP) = E.plot(label='Reference')
E.propagate(0.3).plot(fig, label='30 cm')
E.propagate(0.5).plot(fig, label='50 cm')
E.propagate(0.8).plot(fig, label='80 cm')


axI.set_xlim(-2*R, 2*R)
axI.set_ylabel('Intensity')
axI.margins(0.1)
axP.set_ylim(-0.2, 0.5)
axI.legend()

# plt.figure()
# plt.plot(x*1e3, E.propagate(0.3).I, label='30 cm')
# plt.plot(x*1e3, E.propagate(0.5).I, label='50 cm')
# plt.plot(x*1e3, E.propagate(0.7).I, label='70 cm')
# plt.xlim(-2, 2)
# plt.legend()
# plt.grid()
# plt.xlabel('x (mm)')
# plt.ylabel('Intensity')
plt.show()
