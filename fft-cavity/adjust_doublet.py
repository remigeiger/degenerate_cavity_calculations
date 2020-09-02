import numpy as np
import matplotlib.pyplot as plt
import logging

import fftcavity1d as fc
from efield.efields import hg_efield
import efield

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def propagate_through(interfaces, E_in):
    E = efield.EField.from_EField(E_in)
    for interface in interfaces:
        E = interface.process(E)
    return E

def round_trip(interfaces, E_in, last=True):
    if last:
        round_trip_interfaces = interfaces + interfaces[-1::-1]
    else:
        round_trip_interfaces = interfaces + interfaces[-2:1:-1]
    return propagate_through(round_trip_interfaces, E_in)


def nth_round_trip(interfaces, E_in, n, **rt_kwargs):
    E = efield.EField.from_EField(E_in)
    for _ in range(n):
        E = round_trip(interfaces, E, **rt_kwargs)
    return E


f = 0.2
R = 0.98
TL = 1.0

interfaces = []
# interfaces.append(fc.interfaces.FreeSpace(408.455687e-3))
# interfaces.append(fc.interfaces.ParabolicLens(f=103e-3, T=TL))
# interfaces.append(fc.interfaces.FreeSpace(36.73e-3))
# interfaces.append(fc.interfaces.ParabolicLens(f=-103e-3, T=TL))
# # interfaces.append(fc.interfaces.FreeSpace(181.7988e-3))
# interfaces.append(fc.interfaces.FreeSpace(185.84384e-3))
interfaces.append(fc.interfaces.FreeSpace(253.74e-3))
interfaces.append(fc.interfaces.SphericalLens(f=100e-3, T=TL))
interfaces.append(fc.interfaces.FreeSpace(17e-3))
interfaces.append(fc.interfaces.SphericalLens(f=-100e-3, T=TL))
# interfaces.append(fc.interfaces.FreeSpace(183.81671435e-3 + 2e-3)) # Parabolic lens focus
interfaces.append(fc.interfaces.FreeSpace(179.82e-3))

# interfaces.append(fc.interfaces.FreeSpace(200e-3))
# interfaces.append(fc.interfaces.ParabolicLens(f=200e-3, T=TL))
# interfaces.append(fc.interfaces.FreeSpace(200e-3))




""" Defining the input field """
lda = 852e-9
win = 1e-3
x0, N = 6 * win, 2**14
x = np.linspace(-x0, x0, N)
E_ref = hg_efield(
    n=0, x=x, w=win, amplitude=1.0, x0=0,
    normalize_power=True, lda=lda, radial=False,
    prop_type='evanescent',
)

# E_once = propagate_through(interfaces, E_ref)
fig, ax = plt.subplots()
E_rt = round_trip(interfaces, E_ref)
E_rt2 = round_trip(interfaces, E_rt)
E_rt100 = nth_round_trip(interfaces, E_ref, 100)

fig, (axI, axP) = efield.EField.create_figure()
E_ref.plot(fig, normalize=True, label='ref field')
# E_once.plot(fig, normalize=True, label='once propagated')
E_rt.plot(fig, normalize=True, label='Round trip')
E_rt2.plot(fig, normalize=True, label='Round trip 2')
# nth_round_trip(interfaces, E_ref, 20).plot(fig, normalize=True, label='Round trip 50')
E_rt100.plot(fig, normalize=True, label='Round trip 100')

axI.legend()
# axP.set_ylim(-0.01, 0.01)

plt.show()