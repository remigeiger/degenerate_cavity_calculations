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

N = 2 ** 9
f = 0.20
R = 0.9
TL = 1.0


config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-1878.5e-6)
config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[2.6*45], delta=-45*1878.5e-6 - 330e-6 - 168e-6)

# config = dict(waist=1e-3, orders=[(4, 0)], coeffs=[10.0], delta=0e-6)


finess = np.pi*np.sqrt(R*TL) / (1 - R*TL)
# finess = np.pi * np.sqrt(R) / (1 -
print('Finesse : {:.0f}'.format(finess))


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

def power(rs, field):
    return np.trapz(x=rs, y=rs * abs(field)**2)

def power_at_phase(rs, fields, phase):
    field = total_field(fields, phase)
    return power(rs, field)


lda = 852e-9


w = config['waist']
a = 2.5*w

dx = 2*a/N
print('Condition lentille a 2w : {:.2e} > 1'.format(N * lda * f / (2 * a ** 2)))
print('Condition longue propagation : {:.2e} > 1'.format(N * dx * np.sqrt(4 * dx ** 2 - lda ** 2) / 2 / lda / f))

rs, ps = hankel.rp_space(N, a)
H_Kernel = hankel.kernel(N)


zr = np.pi * w ** 2 / lda
E = np.exp(-rs ** 2 / w ** 2, dtype=np.complex128)
E /= np.sqrt(np.trapz(x=rs, y=rs * abs(E) ** 2))

#E *= np.exp(1j*np.pi/4)

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

aberration = np.exp(1j*2*np.pi*phase_map)
# apo = np.cos(0.5*np.pi*np.arange(N)/N)*0 + 1

fig_spectrum, ax_spectrum = plt.subplots(figsize=(5, 4))
fig_modes, ax_mode = plt.subplots()
images = []
delta_quadratic_compensation = config['delta']

gains = []

d2s = np.linspace(-500e-6, 500e-6, 7) + delta_quadratic_compensation

ret = []
gain_ref = None
phase_ref = None
for d2 in tqdm(d2s):
    propagator1 = propagator(H_Kernel, f, lda, ps)
    propagator2 = propagator(H_Kernel, 2*(f + d2), lda, ps)

    displacement_phase_factor = np.exp(-2*2*np.pi/lda*1j*d2)

    def cavity(E):
        E_p = E*aberration
        E_p = propagator1.dot(E_p)
        E_p = lens(E_p, f)
        E_p = propagator2.dot(E_p)
        E_p = lens(E_p, f)
        E_p = propagator1.dot(E_p)
        E_p *= displacement_phase_factor
        return E_p * R * TL


    fields = get_fields(cavity, np.sqrt(1-R)*E, n_rt=340)

    s = Spectrum(lambda phase: power_at_phase(rs, fields, phase))

    resonance_phases = s.compute_resonances(n_res=1, gain_thres=0.1)


    limits = (resonance_phases[0] - np.pi, resonance_phases[0] + np.pi)

    phases, spectrum = s.spectrum(linspace_kwargs={'focus_width': 1/30, 'num_focused': 300, 'limits': limits})

    ret.append(phases)
    ret.append(spectrum)

    gains.append(spectrum.max())

    if phase_ref is None:
        phase_ref = resonance_phases[0]
    # if gain_ref is None:
    #     gain_ref = spectrum.max()
    # spectrum /= gain_ref

    idx_max = spectrum.argmax()
    freqs = (phases - phase_ref) * 330/2/np.pi

    ax_spectrum.plot(freqs, spectrum, label=r'$\delta_2$ = {:.1f} $\mu$m'.format((d2 - delta_quadratic_compensation)*1e6))
    mode = total_field(fields, resonance_phases[0])


    image = ih.image_from_radial(abs(mode) ** 2, N=100, ratio=0.8)
    images.append(image)

    ax_mode.plot(rs*1e3, abs(mode) ** 2)

gains = np.array(gains)
print(gains[0])

# gains /= gains[0]

filename = 'data/{:03.2f}SA3-{:03.1f}mm-{{}}'.format(config['coeffs'][0], w*1e3)
np.save(filename.format('spectrums'), ret)


fig, ax_gain = plt.subplots()
ax_gain.plot((d2s - delta_quadratic_compensation)*1e6, gains)

ax_spectrum.legend()
ax_spectrum.grid()
ax_spectrum.set_xlabel('Fréquence (MHz)')
ax_spectrum.set_ylabel('Gain optique normalisé')

plt.show()
