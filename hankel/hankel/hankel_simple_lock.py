from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssparse
import scipy.optimize as so
from tqdm import tqdm
import numba

from efield import efields
import hankel
from fftcavity1d.spectrum import Spectrum
import image_helpers as ih
import zernike

N = 2 ** 11


w = 0.25e-3
a = 5*w
rs, ps = hankel.rp_space(N, a)
Y = hankel.kernel(N)
lda = 852e-9
zr = np.pi * w ** 2 / lda

ef = efields.hg_efield(
    1, rs, w,
    lda=lda,
)
ef = efields.lg_efield(
    2, rs, w,
    lda=lda,
)
E = ef.E

# E = np.exp(-rs ** 2 / w ** 2, dtype=np.complex128)

E /= np.sqrt(np.trapz(x=rs, y=rs * abs(E) ** 2))

f = 0.2
R = 0.97
TL = 1.0
d1 = f+12e-3
d2 = f+12e-3

finess = np.pi * np.sqrt(R) / (1 - R)

def lens(E, f):
#    wf = 2*np.pi/lda * f * (1 + 0.5*rs**2 / f**2)
    wf = 2 * np.pi / lda * f * np.sqrt(1 + rs ** 2 / f ** 2)
    return E * np.exp(-1j * wf)


def propagator(kernel, dist, lda, freqs):
    phase_shift_prop = 2 * np.pi / lda * dist + np.pi * lda * dist * freqs ** 2
    return kernel.dot(ssparse.diags(np.exp(-1j * phase_shift_prop)).dot(kernel))


def get_fields(cavity_fun, input_field, n_rt):

    fields = np.zeros((n_rt, len(E)), dtype=np.complex128)
    fields[0, :] = input_field
    for i_rt in range(n_rt - 1):
        fields[i_rt + 1, :] = cavity_fun(fields[i_rt, :])
    return fields


@numba.jit
def total_field(fields, phase):
    shape = fields.shape
    total_field = np.zeros(shape[1], dtype=np.complex128)
    for i in range(shape[0]):
        factor = np.exp(-1j * (i + 1) * phase)
        for k in range(shape[1]):
            total_field[k] = total_field[k] + fields[i, k] * factor
    return total_field


def power(rs, fields, phase):
    field = total_field(fields, phase)
    return np.trapz(x=rs, y=rs * abs(field) ** 2)


fig, ax_spectrum = plt.subplots()
fig.canvas.set_window_title('Spectrum')
fig, ax_mode = plt.subplots()
fig.canvas.set_window_title('Mode')

propagator1 = propagator(Y, d1, lda, ps)
propagator2 = propagator(Y, 2*d2, lda, ps)


def cavity(E):
    E_p = propagator1.dot(E)
    E_p = lens(E_p, f)
    E_p = propagator2.dot(E_p)
    E_p = lens(E_p, f)
    E_p = propagator1.dot(E_p)
    return E_p * R * TL


fields = get_fields(cavity, np.sqrt(1-R)*E, n_rt=250)

s = Spectrum(lambda phase: power(rs, fields, phase))

resonance_phases = s.compute_resonances(n_res=5, gain_thres=1)

phases, spectrum = s.spectrum()
ax_spectrum.plot((phases)*375/(2*np.pi), spectrum)
mode = total_field(fields, resonance_phases[0])


ax_mode.plot(rs*1e3, abs(mode) ** 2)


plt.show()
