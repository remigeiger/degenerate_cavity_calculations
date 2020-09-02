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

N = 2 ** 10 # number of points for the calculation grid
f = 0.228 # Focal length
R = 0.988 # Mirror reflection
#TL = 0.995 # Lens transmission
TL = 0.991 # Lens transmission
lda = 852e-9 # wavelength

finess = np.pi*np.sqrt(R*TL) / (1 - R*TL)
print('Finesse : {:.0f}'.format(finess))

#config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0.13], delta=-145e-6)
#config = dict(waist=5.1e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1850e-6)
#config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1850e-6)

# config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-1870e-6)
# config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-930e-6)
# config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0.65], delta=-465e-6)
# config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0.3], delta=-212.5e-6)
# config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0.1], delta=-70.5e-6)
# config = dict(waist=3.6e-3, orders=[(4, 0)], coeffs=[0], delta=0e-6)

#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-1878e-6)
#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-942e-6) # f=0.200

# config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[2.6], delta=-1862e-6)
# config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[1.3], delta=-925e-6)
# config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[0.65], delta=-460e-6)
# config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[0.3], delta=-210e-6)
# config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[0.1], delta=-69.5e-6)
# config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[0], delta=-0e-6)

# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[2.6], delta=-1893.5e-6)
# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3], delta=-947e-6)
# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[0.65], delta=-473.3e-6)
# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[0.3], delta=-218.4e-6)
# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[0.1], delta=-72.8e-6)
# config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[0], delta=0)

# config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-1862e-6)
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1885e-6)
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-930e-6) # for f=0.200

# *******************************************************************************
# ***** unified settings for the paper after optimization of delta 2 ********
#config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[0], delta=0)
#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[0], delta=0)
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[0], delta=0)

#config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1222e-6) # f=0.228, R = 0.988, TL = 0.995 --> gain = 41.4
#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1217e-6) # f=0.228, R = 0.988, TL = 0.995 --> gain = 40.4
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1205e-6) # f=0.228, R = 0.988, TL = 0.995 --> gain = 26.4

# coeffcient of 1.3 lambda applied on the mirror
#config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1228e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 27.43
#config = dict(waist=2.0e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1220e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 27.25
#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1217e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 26.70
#config = dict(waist=3.2e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1210e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 25.54
#config = dict(waist=3.8e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1210e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 23.97
#config = dict(waist=4.4e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1210e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 21.94
#config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1205e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 20.15
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1205e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 18.15
#config = dict(waist=6.2e-3, orders=[(4, 0)], coeffs=[1.3], delta=-1200e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 16.49

# coeffcient of 2.6 lambda applied on the mirror (effectively 1.3 lambda of the lens)
#config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2450e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 27.33
#config = dict(waist=2.0e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2450e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 26.96
#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2430e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 25.44
#config = dict(waist=3.2e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2445e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 23.40
#config = dict(waist=3.8e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2420e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 21.49
#config = dict(waist=4.4e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2420e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 19.40
#config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2420e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 17.26
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2416e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 15.29
#config = dict(waist=6.2e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2414e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 13.59
#config = dict(waist=6.8e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2411e-6) # f=0.228, R = 0.988, TL = 0.991 --> gain = 12.06

# coeffcient of 2.6 lambda applied on the mirror (effectively 1.3 lambda of the lens), with r_optic = 22.5e-3, f=0.228, R = 0.988, TL = 0.991
#config = dict(waist=0.8e-3, orders=[(4, 0)], coeffs=[2.6], delta=-3055e-6) #  gain = 27.43
#config = dict(waist=1.4e-3, orders=[(4, 0)], coeffs=[2.6], delta=-3030e-6) #  gain = 27.34
#config = dict(waist=2.0e-3, orders=[(4, 0)], coeffs=[2.6], delta=-3030e-6) #  gain = 26.36
#config = dict(waist=2.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-3005e-6) #  gain = 24.98
#config = dict(waist=3.2e-3, orders=[(4, 0)], coeffs=[2.6], delta=-3005e-6) #  gain = 22.59
#config = dict(waist=3.8e-3, orders=[(4, 0)], coeffs=[2.6], delta=-3005e-6) #  gain = 19.77
#config = dict(waist=4.4e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2980e-6) #  gain = 17.51
#config = dict(waist=5.0e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2980e-6) #  gain = 15.39
#config = dict(waist=5.6e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2980e-6) #  gain = 13.48
#config = dict(waist=6.2e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2980e-6) #  gain = 11.81
#config = dict(waist=6.8e-3, orders=[(4, 0)], coeffs=[2.6], delta=-2975e-6) #  gain = 10.46

# vary the coeff of aberration around 2.6 with r_optic = 45 mm to see the dependence of the gain
#config = dict(waist=6.8e-3, orders=[(4, 0)], coeffs=[2.4], delta=-2700e-6) #  gain = 10.67
#config = dict(waist=6.8e-3, orders=[(4, 0)], coeffs=[2.8], delta=-3200e-6) #  gain = 10.17


# ********* functions ************
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
    total_field = np.zeros(shape[1], dtype=np.complex128) # dimension N_x
    for i in range(shape[0]): # fields = matrix of size Nrt x N_x
        # factor = np.cos((1+i)*phase) + 1j*np.sin((1+i)*phase)
        factor = np.exp(-1j * (i + 1) * phase)
        for k in range(shape[1]): # boucle sur la dimension spatiale, équivalente à une écriture plus simple en python mais optimisée avec @numba.jit
            total_field[k] = total_field[k] + fields[i, k] * factor
    return total_field

def power(rs, field):
    return np.trapz(x=rs, y=rs * abs(field)**2)

def power_at_phase(rs, fields, phase):
    field = total_field(fields, phase)
    return power(rs, field)

def fermidirac(x, amplitude, R0, x0, b):
    e = np.exp(-b * (abs(x - x0) / R0 - 1))
    return amplitude * e / (1 + e)


# ********** start of the main programme *******


w = config['waist']
a = 3*w # maximal radius value

dx = 2*a/N
print('Condition lentille a 2w : {:.2e} > 1'.format(N * lda * f / (2 * a ** 2)))
print('Condition longue propagation : {:.2e} > 1'.format(N * dx * np.sqrt(4 * dx ** 2 - lda ** 2) / 2 / lda / f))

rs, ps = hankel.rp_space(N, a)
H_Kernel = hankel.kernel(N)


zr = np.pi * w ** 2 / lda
E = np.exp(-rs ** 2 / w ** 2, dtype=np.complex128)
E /= np.sqrt(np.trapz(x=rs, y=rs * abs(E) ** 2))


r_optic = 22.5e-3
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

# span of distance variation delta2 to find the best resonance ;
d2s = np.linspace(-0e-6, -150e-6, 7) + delta_quadratic_compensation
#d2s = np.linspace(800e-6, 1200e-6, 20) + delta_quadratic_compensation

ret = []
gain_ref = None
phase_ref = None
for d2 in tqdm(d2s):
    propagator1 = propagator(H_Kernel, f, lda, ps)
    propagator2 = propagator(H_Kernel, 2*(f + d2), lda, ps) # propagator lens - small mirror - lens

    # removes the propagation phase corresponding to the displacement of delta2 in order to keep resonances centered around f = 0 Hz in the plot 
    displacement_phase_factor = np.exp(-2*2*np.pi/lda*1j*d2) 

    # the definition of the cavity function is in the loop as it depends on delta2
    def cavity(E):
        E_p = E*aberration
        E_p = propagator1.dot(E_p)
        E_p = lens(E_p, f)
        E_p = propagator2.dot(E_p)
        E_p = lens(E_p, f)
        E_p = propagator1.dot(E_p)
        E_p *= displacement_phase_factor
        return E_p * R * TL


    fields = get_fields(cavity, np.sqrt(1-R)*E, n_rt=340) # vérifier avec 2*finesse

    s = Spectrum(lambda phase: power_at_phase(rs, fields, phase)) # helps to find the resonance of the cavity

    resonance_phases = s.compute_resonances(n_res=1, gain_thres=0.1)


    limits = (resonance_phases[0] - np.pi, resonance_phases[0] + np.pi)

    # to plot the spectrum,  we vary a phase but the resonane might be badly sampled 
    # this function creates an adequate linspace to have many points around the resonance
    # focus width = width in FSR around the resonance where we want many points
    # num focus = how many points in this interval
    # limits = range of the scan
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
    freqs = (phases - phase_ref) * 330/2/np.pi  # 330 = ISL c/2L = c/4f with f = 0.228 m ;

    # figure to show the spectra for different values of delta2
    ax_spectrum.plot(freqs, spectrum, label=r'$\delta_2$ = {:.1f} $\mu$m'.format((d2 - delta_quadratic_compensation)*1e6))
    mode = total_field(fields, resonance_phases[0])


    image = ih.image_from_radial(abs(mode) ** 2, N=100, ratio=0.8)
    images.append(image)

    ax_mode.plot(rs*1e3, abs(mode) ** 2)

gains = np.array(gains)
print('maximum optical gain = {:2.2f}'.format(gains[np.argmax(gains)]))
print('corresponding value of delta_2 = {:2.2f} microns'.format(1e6*(d2s[np.argmax(gains)])))

# gains /= gains[0]

filename = 'data/{:03.2f}SA3-{:03.1f}mm-{{}}'.format(config['coeffs'][0], w*1e3)
np.save(filename.format('spectrums'), ret)

# figure to show the maximal gain found for different values of delta2
fig, ax_gain = plt.subplots()
ax_gain.plot((d2s - delta_quadratic_compensation)*1e6, gains,'.--')
ax_gain.set_xlabel('$\delta_2 ($\mu m$)$')
ax_gain.set_ylabel('optical gain')
ax_gain.grid()

ax_spectrum.legend()
ax_spectrum.grid()
ax_spectrum.set_xlabel('Frequency (MHz)')
ax_spectrum.set_ylabel('Normalized optical gain')

plt.show()
