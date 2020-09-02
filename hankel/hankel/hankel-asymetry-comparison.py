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

N = 2 ** 10
f = 0.225
R = 0.988
TL = 0.995

config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0.13*2*np.pi], delta=-145e-6)
config = dict(waist=5.1e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1850e-6)
config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1850e-6)
# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1875e-6)
# config = dict(waist=1.4e-3, orders=[], coeffs=[], delta=0)
# config = dict(waist=2.6e-3, orders=[], coeffs=[], delta=0)
# config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[0.1*2*np.pi], delta=-166.7e-6-16.7e-6+6.7e-6)
# config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[0.65*2*np.pi], delta=-1166.7e-6)
# config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1800e-6-500e-6-35e-6)
# config = dict(waist=4e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1800e-6-500e-6-35e-6-6.7e-6)
# config = dict(waist=1.4e-3, orders=[], coeffs=[], delta=0)

config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-2346.3e-6)


finess = np.pi * np.sqrt(R) / (1 - R)


def lens(E, f):
    # wf = 2*np.pi/lda * f * (1 + 0.5*rs**2 / f**2)
    wf = 2 * np.pi / lda * f * np.sqrt(1 + rs ** 2 / f ** 2) # - 2*np.pi/lda*f
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


lda = 852e-9

w = config['waist']
a = 4*w

dx = 2*a/N

print('Condition lentille a 2w : {:.2e} > 1'.format(N*lda*f/(2*a**2)))
print('Condition longue propagation : {:.2e} > 1'.format(N*dx*np.sqrt(4*dx**2 - lda**2)/2/lda/f))

# exit()

rs, ps = hankel.rp_space(N, a)
H_Kernel = hankel.kernel(N)


zr = np.pi * w ** 2 / lda
E = np.exp(-rs ** 2 / w ** 2, dtype=np.complex128)
E /= np.sqrt(np.trapz(x=rs, y=rs * abs(E) ** 2))



r_optic = 25e-3
orders = config['orders']
coeffs = config['coeffs']
phase_map = zernike.zernike_combination(
    orders,
    np.array(coeffs),
    rs / r_optic,
    phi=0.0
)

#plt.figure()
#plt.plot(rs*1e3, phase_map/(2*np.pi))
# cavity.insert(1, PhaseMask(phase_map))

# phase_map = generate_map(len(x), 0.2) * 0.025 * 2 * np.pi
# cavity.insert(1, PhaseMask(phase_map))
# cavity.insert(3, PhaseMask(phase_map))

aberration = np.exp(1j*phase_map)
# apo = np.cos(0.5*np.pi*np.arange(N)/N)*0 + 1

ref_max = None

fig_spectrum, ax_spectrum = plt.subplots(figsize=(5, 4))
fig, ax_mode = plt.subplots()
images = []
delta_quadratic_compensation = config['delta']

d2s = np.array([0, 25, 50, 75, 100, 130, 200])*1e-6
d2s = np.linspace(0, 150e-6, 7)

for d2 in tqdm(d2s + delta_quadratic_compensation):
    propagator1 = propagator(H_Kernel, f, lda, ps)
    propagator2 = propagator(H_Kernel, 2*(f + d2), lda, ps)

    displacement_phase_factor = np.exp(-2*2*np.pi/lda*1j*d2 + 1j*np.pi)

    def cavity(E):
        E_p = propagator1.dot(E)
        # E_p = propagator1.dot(E_p)
        E_p = lens(E_p, f)
        E_p *= aberration
        # E_p = propagator2.dot(E_p)
        # E_p = propagator1.dot(E_p)
        # E_p = propagator2.dot(E_p)
        E_p = propagator2.dot(E_p)
        E_p *= displacement_phase_factor
        E_p = lens(E_p, f)
        E_p *= aberration
        E_p = propagator1.dot(E_p)
        # E_p = propagator1.dot(E_p)
#        E_p *= apo
        return E_p * R * TL


    fields = get_fields(cavity, np.sqrt(1-R)*E, n_rt=400)

    s = Spectrum(lambda phase: power(rs, fields, phase))

    resonance_phases = s.compute_resonances(n_res=1, gain_thres=0.1)

    phases, spectrum = s.spectrum(linspace_kwargs={'focus_width': 1/15, 'num_focused': 300})

    if ref_max is None:
        ref_max = spectrum.max()
    # spectrum /= spectrum.max()
    idx_max = spectrum.argmax()

    freqs = (phases-phases[idx_max]) * 330/2/np.pi
    freqs = phases * 330 / 2 /np.pi

    label ='{:.1f}'.format((d2 - delta_quadratic_compensation)*1e6)
    # label = ''

    ax_spectrum.plot(freqs, spectrum / ref_max, label=label)
    mode = total_field(fields, resonance_phases[0])

    image = ih.image_from_radial(abs(mode) ** 2, N=100, ratio=0.8)
    images.append(image)

    ax_mode.plot(rs*1e3, abs(mode) ** 2)

big_img = ih.stack_images(images, n_per_row=5)
fig, ax = plt.subplots()
ax.imshow(big_img, cmap='gray')

ax_spectrum.legend()
ax_spectrum.grid()
ax_spectrum.set_xlabel('Fréquence (MHz)')
ax_spectrum.set_ylabel('Gain optique normalisé')

fig_spectrum.tight_layout()

plt.show()
