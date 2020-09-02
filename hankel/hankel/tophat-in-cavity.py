from time import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssparse
import scipy.optimize as so
from tqdm import tqdm
import numba

import hankel
from fftcavity1d.spectrum import Spectrum
import image_helpers as ih
import zernike

import datetime

N = 2 ** 10
f = 0.225
R = 0.988
TL = 0.995

finess = np.pi * np.sqrt(R) / (1 - R)

def lens(E, f):
    # wf = 2*np.pi/lda * f * (1 + 0.5*rs**2 / f**2)
    wf = 2 * np.pi / lda * f * np.sqrt(1 + rs ** 2 / f ** 2)
    return E * np.exp(-1j * wf)


def propagator(kernel, dist, lda, freqs):
    # phase_shift_prop = 2 * np.pi / lda * dist + np.pi * lda * dist * freqs ** 2
    phase_shift_prop = -2*np.pi / lda * dist * np.sqrt(1 - lda ** 2 * freqs ** 2 + 0j)
    return kernel.dot(ssparse.diags(np.exp(-1j * phase_shift_prop)).dot(kernel))


def get_fields(cavity_fun, input_field, n_rt):

    fields = np.zeros((n_rt, len(E)), dtype=np.complex128)
    fields[0, :] = input_field
    for i_rt in range(n_rt - 1):
        fields[i_rt + 1, :] = cavity_fun(fields[i_rt, :])
    return fields


@numba.jit
def total_field(fields, phase):
    # factor = np.exp(1j*np.arange(fields.shape[0])*phase)
    # return np.einsum('ji,j->i', fields, factor)

    shape = fields.shape
    total_field = np.zeros(shape[1], dtype=np.complex128)
    for i in range(shape[0]):
        # factor = np.cos((1+i)*phase) + 1j*np.sin((1+i)*phase)
        factor = np.exp(-1j * (i + 1) * phase)
        for k in range(shape[1]):
            total_field[k] = total_field[k] + fields[i, k] * factor
    return total_field


def power(rs, fields, phase):
    field = total_field(fields, phase)
    return np.trapz(x=rs, y=rs * abs(field) ** 2)

def fermi_dirac(r, radius, beta, amplitude):
    E = np.exp(-beta*(r/radius - 1))
    return amplitude * E/(1+E)


lda = 852e-9


fd_radius = 3.6e-3
a = 3*fd_radius

dx = 2*a/N

print('Condition lentille a 2w : {:.2e} > 1'.format(N*lda*f/(2*a**2)))
print('Condition longue propagation : {:.2e} > 1'.format(N*dx*np.sqrt(4*dx**2 - lda**2)/2/lda/f))


rs, ps = hankel.rp_space(N, a)
H_Kernel = hankel.kernel(N)


E = fermi_dirac(rs, fd_radius, 14, 1.0).astype(np.complex128)
E /= np.sqrt(np.trapz(x=rs, y=rs * abs(E) ** 2))

#E = np.exp(-rs ** 2 / fd_radius ** 2, dtype=np.complex128)
#E /= np.sqrt(np.trapz(x=rs, y=rs * abs(E) ** 2))


fig_spectrum, ax_spectrum = plt.subplots(figsize=(5, 3))
fig_mode, ax_mode = plt.subplots(figsize=(5, 3))
images = []
delta_quadratic_compensation = 0

phase_ref = None
intensity_ref = None

for d2 in tqdm(np.linspace(0, 50e-6, 3) + delta_quadratic_compensation):
    propagator1 = propagator(H_Kernel, f, lda, ps)
    propagator2 = propagator(H_Kernel, 2*(f - d2), lda, ps)

    displacement_phase_factor = np.exp(2*2*np.pi/lda*1j*d2)

    def cavity(E):
        E_p = propagator1.dot(E)
        E_p = lens(E_p, f)
        E_p = propagator2.dot(E_p)
        E_p *= displacement_phase_factor
        E_p = lens(E_p, f)
        E_p = propagator1.dot(E_p)
        return E_p * R * TL


    fields = get_fields(cavity, np.sqrt(1-R)*E, n_rt=300)

    s = Spectrum(lambda phase: power(rs, fields, phase))

    resonance_phases = s.compute_resonances(n_res=5, gain_thres=1)

    if phase_ref is None:
        phase_ref = resonance_phases[0]

    phases, spectrum = s.spectrum(linspace_kwargs={'focus_width': 1/30})
    ax_spectrum.plot((phases - phase_ref)/(2*np.pi), spectrum, label=r'$\delta_2$= {:.1f} $\mu$m'.format((d2 - delta_quadratic_compensation)*1e6))
    mode = total_field(fields, resonance_phases[0])
    if intensity_ref is None:
        intensity_ref = (abs(mode)**2).max()

    image = ih.image_from_radial(abs(mode) ** 2, N=100, ratio=0.8)
    images.append(image)

    ax_mode.plot(rs*1e3, abs(mode) ** 2 / intensity_ref)

ax_mode.grid()
ax_mode.set_xlabel('r (mm)')
ax_mode.set_ylabel('Intensité (u.a.)')
ax_spectrum.grid()
ax_spectrum.set_xlabel('Fréquence normalisée')
ax_spectrum.set_ylabel('Gain optique')
ax_spectrum.legend()

fig_spectrum.tight_layout()
fig_mode.tight_layout()

big_img = ih.stack_images(images, n_per_row=3)
fig, ax = plt.subplots()
ax.imshow(big_img, cmap='gray')

plt.show()
