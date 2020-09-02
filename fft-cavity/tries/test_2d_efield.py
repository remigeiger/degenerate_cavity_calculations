import numpy as np
import matplotlib.pyplot as plt

from efield.efield2d import EField2d
import fftcavity1d as fc


nx = 2**10
ny = 2**9

w = 0.5e-3
lda = 852e-9

x = np.linspace(-5*w, 5*w, nx)
y = np.linspace(-5*w, 5*w, ny)

X, Y = np.meshgrid(x, y)

E = np.exp(-(X**2 + Y**2)/w**2)

E_in = EField2d(
    x=x, y=y, E=E, lda=852e-9,
    normalize_power=False, prop_type='fresnel',
)

plt.figure()
plt.plot(E_in.I[ny//2, :])

E_out = E_in.propagated(1)
plt.plot(E_out.I[ny//2, :])


f = 0.2

cavity = fc.cavities.MIGACavity(f=f, d1=f, d2=f,
                                R1=0.98, R2=0.98, TL=1.0,
                                parabolic_lens=False)

handler = fc.CavityHandler(cavity, radial=False)
n_roundtrips = 3 # int(2 * cavity.finess)

handler.calculate_fields(E_in, N=n_roundtrips)

plt.show()