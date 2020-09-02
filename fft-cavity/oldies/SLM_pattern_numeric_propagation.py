from __future__ import division

import numpy as np
# import scipy.fftpack as sf
from math import pi
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from beam_shaper.BeamShaper import BeamShaper
from EField import EField

sns.set_style('white')


def gaussian(x, amplitude, center, waist, offset):
    return amplitude * np.exp(-(x-center)**2/waist**2) + offset


""" Input Field """
win = 1.06e-3
N = 2**12
x0 = 25*win
x = np.linspace(-x0, x0, N)

lda = 770e-9
E_ref = EField(x, E=gaussian(x, amplitude=1, center=0, waist=win, offset=0),
               lda=lda)

""" Beam Shaper """
keplerian = True

w0 = 1.06e-3
gauss_dist = lambda r: np.exp(-2*r**2/w0**2)

R0 = 1.3e-3
b = 16.8

out_dist = lambda r: 1/(1 + np.exp(b*(abs(r)/R0 - 1)))
# out_dist = lambda r: np.exp(-2*r**2/R0**2)

n = 2
d = 300e-3
BS = BeamShaper(gauss_dist, out_dist, d, n, keplerian=keplerian)

z1 = BS.first_lens(abs(x))
e1 = BS.first_lens(BS.r0)
z2 = BS.second_lens(abs(x))
e2 = BS.second_lens(BS.R0)


phase_1 = E_ref.k*(n-1)*(e1-z1)
phase_2 = E_ref.k*(n-1)*(e2-z2)


# plt.figure()
# plt.plot(phase_1)
# plt.plot(phase_2)
# plt.show()

phase_1[np.isnan(phase_1)] = 0
phase_2[np.isnan(phase_2)] = 0
if keplerian:
    phase_1 = -phase_1
    phase_2 = phase_2

""" Calculation """
E_trans = E_ref.propagate(-1.8).transmit(np.exp(-1j*phase_1))
dist = 33e-2
E_prop = E_trans.propagate(dist)
E_prop.E[abs(x) > 1.5*win] = 0
E_trans2 = E_prop.transmit(np.exp(-1j*phase_2))
E_prop2 = E_trans2.propagate(0.68)

pickle.dump([E_prop2.x, E_prop2.I], open('680.p', 'w'))

""" """
# fig, (axI, axP) = E_ref.plot(label='Reference')
# axI.set_xlim(-3*win, 3*win)
# axP.set_ylim(0, 10)
# # E_ref.propagate(3).plot(fig, label='test')
# E_prop.plot(fig, label='Modulated and Propagated')
# E_trans2.plot(fig, intensity=False, label='Remodulated')
# E_prop2.plot(fig, label='Repropagated')
# # E_ref.propagate(d).plot(fig)
#
# axI.plot(x, out_dist(x)*E_prop.I.max(), label='Theoretic goal')
# axI.margins(0.1)
#
# axP.plot(x, (n-1)*z2*E_ref.k/pi, label='SLM Phase')
#
# # axP.plot(x, -phase_2/pi)
# axI.legend()
# axP.legend()
# plt.show()

""" """
# fig, (axI, axP) = E_ref.plot(label='Reference')
# axI.set_xlim(-3*win, 3*win)
# axP.set_ylim(0, 10)
# plt.figure()
# plt.plot(x*1e3, out_dist(x)*E_prop.I.max(), label='Theoretic goal')
# for dist in [33e-2, 49e-2, 68e-2]:
#     E = E_trans.propagate(dist)
#     plt.plot(x*1e3, E.I, label='{} cm'.format(dist*1e2))
#     pickle.dump([E.x, E.I], open('{:.0f}.p'.format(dist*1e3), 'w'))
#
# plt.xlim(-2, 2)
# plt.legend()
# plt.show()
