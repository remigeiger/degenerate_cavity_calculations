import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as ssparse
import numba

import hankel
from fftcavity1d.spectrum import Spectrum
import image_helpers as ih
import zernike

N = 2 ** 11
f = 0.2
R = 0.98
TL = 1.0

config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0.13*2*np.pi], delta=-145e-6)
config = dict(waist=5.1e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1850e-6)
config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1850e-6)
config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1875e-6)
# config = dict(waist=3.6e-3, orders=[], coeffs=[], delta=0)

configs = [
    dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3 * 2 * np.pi], delta=-1875e-6),
    dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[1.3 * 2 * np.pi], delta=-1850e-6),
    dict(waist=5.1e-3, orders=[(4, 0)], coeffs=[1.3 * 2 * np.pi], delta=-1850e-6),
]

# configs = [
#     dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3*2*np.pi], delta=-1875e-6),
# ]

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

lda = 852e-9

fig, ax_spectrum = plt.subplots()
fig, ax_mode = plt.subplots()

for config in configs :



    w = config['waist']
    a = 4*w

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


    aberration = np.exp(1j*phase_map)

    images = []
    delta_quadratic_compensation = config['delta']

    propagator1 = propagator(H_Kernel, f, lda, ps)
    propagator2 = propagator(H_Kernel, 2*(f+delta_quadratic_compensation), lda, ps)


    def cavity(E):
        E_p = propagator1.dot(E)
        E_p = lens(E_p, f)
        E_p *= aberration
        E_p = propagator2.dot(E_p)
        E_p = lens(E_p, f)
        E_p *= aberration
        E_p = propagator1.dot(E_p)
        return E_p * R * TL


    fields = get_fields(cavity, np.sqrt(1-R)*E, n_rt=250)

    s = Spectrum(lambda phase: power(rs, fields, phase))

    resonance_phases = s.compute_resonances(n_res=1, gain_thres=1)


    phases, spectrum = s.spectrum(linspace_kwargs={'focus_width': 1/20})
    ax_spectrum.plot((phases - resonance_phases[0])/(2*np.pi), spectrum, label='2w={:.1f}'.format(2*w*1e3))
    mode = total_field(fields, resonance_phases[0])


    image = ih.image_from_radial(abs(mode) ** 2, N=100, ratio=0.8)
    images.append(image)

    I = abs(mode)**2
    ax_mode.plot(rs/w, I / I.max())

ax_spectrum.legend()

big_img = ih.stack_images(images, n_per_row=3)
fig, ax = plt.subplots()
ax.imshow(big_img, cmap='gray')
ax_spectrum.legend()

plt.show()
