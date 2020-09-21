# -*- coding: utf-8 -*-
"""
For given properties of the optics (i.e. aberrations) and given input field, 
calculate the field propagating inside the degenerate cavity with the propagation 
with magnification method. Calculate the associated spectrum of the cavity,
gain and finesse.
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nf
from tqdm import tqdm
import time

import matplotlib.patches as mpatches
import sys
import os
# Adapt your path please
sys.path.insert(0, '/home/constance/Bureau/obsp/codes_article/hankel')
sys.path.insert(0, '/home/constance/Bureau/obsp/codes_article/fftcavity1d')
from spectrum import Spectrum
from zernike import zernike_combination

# ******** global parameters ********
N = 2**9        # number of points for the calculation grid
f = 0.228       # focal length
R = 0.988       # mirror reflection
TL = 0.991      # lens transmission
lda = 852e-9    # wavelength

k0 = 2*np.pi/lda

# ******** configs ********

# *** aberration coefficients ***
ast45_lda = 0.06 
ast0_lda = 0.027
coma45_lda = 0.155
coma0_lda = 0.138
sa3_lda = 1.3
coeffs_all = [ast0_lda, ast45_lda, sa3_lda, coma0_lda, coma45_lda]
orders_all = [(2, +2), (2, -2), (4, 0), (3, 1), (3, -1)]

# *** gaussian configs ***
# *** spherical aberrations only ***
#config = dict(FD = False, waist=0.8e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3055e-6)
#config = dict(FD = False, waist=1.4e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3030e-6)
#config = dict(FD = False, waist=2.0e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3030e-6)
#config = dict(FD = False, waist=2.6e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3005e-6)
#config = dict(FD = False, waist=3.2e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3005e-6)
#config = dict(FD = False, waist=3.8e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3005e-6)
#config = dict(FD = False, waist=4.4e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=5.0e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=5.6e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=6.2e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=6.8e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=2975e-6)

# *** spherical aberrations + astigmatism ***
#config = dict(FD = False, waist=0.8e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=3055e-6)
#config = dict(FD = False, waist=1.4e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=3030e-6)
#config = dict(FD = False, waist=2.0e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=3030e-6)
#config = dict(FD = False, waist=2.6e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=3005e-6)
#config = dict(FD = False, waist=3.2e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=3005e-6)
#config = dict(FD = False, waist=3.8e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=3005e-6)
#config = dict(FD = False, waist=4.4e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=5.0e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=5.6e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=6.2e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=2980e-6)
#config = dict(FD = False, waist=6.8e-3, orders=[(2, +2), (2, -2), (4, 0)], coeffs=[ast0_lda, ast45_lda, sa3_lda], delta=2975e-6)

# *** spherical aberrations + astigmatism + coma ***
#config = dict(FD = False, waist=0.8e-3, orders=orders_all, coeffs=coeffs_all, delta=3055e-6)
#config = dict(FD = False, waist=1.4e-3, orders=orders_all, coeffs=coeffs_all, delta=3030e-6)
#config = dict(FD = False, waist=2.0e-3, orders=orders_all, coeffs=coeffs_all, delta=3030e-6)
#config = dict(FD = False, waist=2.6e-3, orders=orders_all, coeffs=coeffs_all, delta=3005e-6)
#config = dict(FD = False, waist=3.2e-3, orders=orders_all, coeffs=coeffs_all, delta=3005e-6)
#config = dict(FD = False, waist=3.8e-3, orders=orders_all, coeffs=coeffs_all, delta=3005e-6)
#config = dict(FD = False, waist=4.4e-3, orders=orders_all, coeffs=coeffs_all, delta=2980e-6)
#config = dict(FD = False, waist=5.0e-3, orders=orders_all, coeffs=coeffs_all, delta=2980e-6)
#config = dict(FD = False, waist=5.6e-3, orders=orders_all, coeffs=coeffs_all, delta=2980e-6)
#config = dict(FD = False, waist=6.2e-3, orders=orders_all, coeffs=coeffs_all, delta=2980e-6)
#config = dict(FD = False, waist=6.8e-3, orders=orders_all, coeffs=coeffs_all, delta=2975e-6)


# *** Fermi-Dirac configs ***
#config = dict(FD = True, R0=5e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3015e-6)
#config = dict(FD = True, R0=5e-3,orders=orders_all, coeffs=coeffs_all, delta=3015e-6)
#config = dict(FD = True, R0=3e-3, orders=[(4, 0)], coeffs=[sa3_lda], delta=3015e-6)
config = dict(FD = True, R0=3e-3, orders=orders_all, coeffs=coeffs_all, delta=3015e-6)

FD = config['FD']
if FD:
    R0 = config['R0']   # beam radius for Fermi-Dirac beam
    w = R0              # easier for the propagation algorithm afterward
    b = 17
    a = 1.5*w   # maximum radius value
else:
    w = config['waist'] # beam waist for gaussian beam
    a = 3*w   # maximum radius value
dx = 2*a/N


# ******** functions ********
def fermidirac(x, amplitude, R0, x0, b):
    e = np.exp(-b * (abs(x - x0) / R0 - 1))
    return amplitude * e / (1 + e)

def gaussian(X, Y, amplitude, waist, x0=0, y0=0):
    return amplitude * np.exp(-((X-x0)**2 + (Y-y0)**2)/waist**2)

def get_prop_phase_shift(dist, lda, Freqs):
    return 2*np.pi/lda * dist - np.pi * lda * dist * (Freqs[0]**2 + Freqs[1]**2)

def get_lens_wf(f, lda, X, Y):
    wf = 2*np.pi/lda * f * (1 + 0.5 * (X**2 + Y**2) / f ** 2)
    return wf

def propagate(field, prop_factor):
    """
    Propagation function with the angular spectrum method
    """
    return nf.ifft2(nf.fft2(field) * prop_factor)


def prop_mag(field, w0, w1, L, lda, x, y, Freqs):
    """
    Propagation function withe the propagation with magnification method
    Algorithm detailed in 6. Appendix: propagation with magnification
    """
    X, Y = np.meshgrid(x, y)
    delta_z = w0 * L  / w1 
    z0 = L / (w1 / w0 - 1) 
    k = 2 * np.pi / lda

    v = field * np.exp(1j * (k * (X**2 + Y**2) / (2*z0))) # virtual collimation in (x',y') space
    v = nf.fft2(v)
    v = v * np.exp(1j * ((2*np.pi*Freqs[0])**2 + (2*np.pi*Freqs[1])**2) * delta_z / (2*k)) * np.exp(-1j * k * L) # propagation
    v = nf.ifft2(v)
    v = v * (w0 / w1) * np.exp(-1j * k * (X**2 + Y**2) * (z0 + L) / (2 * z0**2)) # virtual de-collimation in (x',y') space

    x = x * w1 / w0 # back to (x,y) space
    y = y * w1 / w0
    fx = nf.fftfreq(len(x), x[1] - x[0]) # adapt the frequency space
    fy = nf.fftfreq(len(y), y[1] - y[0])
    X, Y = np.meshgrid(x, y)
    Freqs = np.meshgrid(fx, fy)

    return v, x, y, Freqs

def get_fields(field, cavity_fun, n_rt):
    fields = np.empty((*field.shape, n_rt), np.complex64)
    fields[:, :, 0] = field
    for num in tqdm(range(n_rt - 1)):
        fields[:, :, num + 1] = cavity_fun(fields[:, :, num])
    return fields

def get_total_field(fields, phase):
    phases = np.exp(1j*np.arange(fields.shape[2])*phase)
    return np.einsum('ijk,k->ij', fields, phases)

def power(field):
    return np.trapz(np.trapz(abs(field)**2))

def power_at_phase(fields, phase):
    total_field = get_total_field(fields, phase)
    return power(total_field)

# *** start of the main program ***
x = np.linspace(-a, a, N)
y = np.linspace(-a, a, N)
X, Y = np.meshgrid(x, y)
freqsx = nf.fftfreq(N, x[1] - x[0])
freqsy = nf.fftfreq(N, y[1] - y[0])
Freqs = np.meshgrid(freqsx, freqsy)

orders = config['orders']
coeffs = config['coeffs']
r_optic = 22.5e-3
phi = np.arctan2(X, Y)
aberration_map = zernike_combination(orders, coeffs, np.sqrt(X**2 + Y**2)/r_optic, phi)

delta = config['delta']
d1 = f          # distance between big mirror and lens
d2 = f + delta  # distance between lens and small mirror

def MIGA_cavity_fun(d1, d2, f, R, TL, FD):
    lens_wf = 2*np.pi/lda * f * (1 + 0.5 * (X**2/f**2 + Y**2/f**2))
    lens_factor = np.exp(1j*lens_wf)
    aberr_factor = np.exp(1j*2*np.pi*aberration_map)
    def cavity_fun(in_field):
        w0 = w                              # waist on the big mirror
        zR = np.pi * w0**2 / lda
        w1 = w0 * np.sqrt(1 + (d1 / zR)**2) # waist on the lens
        w2 = w0 * f / zR                    # waist on the small mirror for delta = 0
        zR2 = np.pi * w2**2 / lda
        w2 = w2 * np.sqrt(1 + (12*(d2 - f) / zR2)**2) # estimation of the waist in the small mirror (slighly overestimated)
        if FD:
            w2 = 1*w2 # estimation of the waist on the small mirror for a flattop beam - estimation from numerical simulations

        out, Xx, Yy, Freqs1 = prop_mag(in_field, w0, w1, d1, lda, x, y, Freqs)
        out = out * lens_factor
        out, Xx, Yy, Freqs1 = prop_mag(out, w1, w2, d2, lda, x, y, Freqs)
        # Observation done with one round trip: the phase is a bit bent (not flat on the big mirror when the cavity
        # is aligned) and after hundreds of round trips, this error on the phase distorts the field a bit.
        # Empirical observations have shown that doing the preceeding trick allows one to recover a non-distorted
        # phase and then yields the correct propagated field. 
        out, Xx, Yy, Freqs1 = prop_mag(out, w2, w1, d2, lda, Xx, Yy, Freqs1)
        out = out * lens_factor
        out, Xx, Yy, Freqs1 = prop_mag(out, w1, w0, d1, lda, Xx, Yy, Freqs1)
        out = out * aberr_factor
        out = out * aberr_factor
        return out * R * TL
    return cavity_fun


cavity_fun = MIGA_cavity_fun(d1, d2, f, R, TL, FD)

if FD:
    in_field = fermidirac(np.sqrt(X**2 + Y**2), 1.0, R0, 0, b).astype(np.complex64)
else:    
    in_field = gaussian(X, Y, 1.0, w, x0=0, y0=0).astype(np.complex64)
in_field /= np.sqrt(power(in_field))

ISL = 3e8 / (2*(d1 + d2)) # Free Spectral Range

print('Computing fields.')
fields = get_fields(np.sqrt(1-R)*in_field, cavity_fun, 50) # take at least 2*finesse
print('Fields computed.')
print('Memory used : {:.0f} Mb'.format(fields.nbytes / 1e6))

r = N//2**7
reduced_fields = np.copy(fields[::r, ::r, :]) * r                   # We reduce the number of points, so renormalize
s = Spectrum(lambda phase: power_at_phase(reduced_fields, phase))   # faster to compute the spectrum than using 
                                                                    # all the preceeding points

print('Computing resonances.')
resonance_phases = s.compute_resonances(n_res=1, gain_thres=0.3)
print('Resonances computed.')

limits = (resonance_phases[0] - np.pi, resonance_phases[0] + np.pi)
phases, spectrum = s.spectrum()
print('resonances = ', resonance_phases)

# *** Plot some cuts of the cavity fields ***
plt.rcParams['figure.figsize'] = (14.0, 6.0)
plt.rcParams['font.size'] = 18
def plot_field(field, axis=0, axI=None, axP=None, **kwargs):
    if axis == 0:
        s = np.s_[:, N//2]
    else:
        s = np.s_[N//2, :]
    if axI is not None:
        axI.plot(abs(field[s])**2, **kwargs)
    if axP is not None:
        phase = np.unwrap(np.angle(field[s]))
        axP.plot(phase - phase[N//2], **kwargs)

fig_x, (axI_x, axP_x) = plt.subplots(2, 1, sharex=True)
fig_x.canvas.set_window_title('x profile')
fig_y, (axI_y, axP_y) = plt.subplots(2, 1, sharex=True)
fig_y.canvas.set_window_title('y profile')
for num in range(0, fields.shape[2], 2*(fields.shape[2]//10)+1):
    plot_field(fields[:, :, num], axis=0, axI=axI_x, axP=axP_x, label='{}'.format(num))
    plot_field(fields[:, :, num], axis=1, axI=axI_y, axP=axP_y, label='{}'.format(num))
axI_x.legend()
axP_x.set_ylim(-0.5, 0.5)
axI_y.legend()
axP_y.set_ylim(-0.5, 0.5)
axI_x.set(title="X direction cut")
axI_y.set(title="Y direction cut")

#*** plot the spectrum ***
fig_spec, ax_spec = plt.subplots()
fig_spec.canvas.set_window_title("Spectrum")
freqs = phases * ISL/2/np.pi
ax_spec.plot(freqs, spectrum, color='k')
yl = ax_spec.get_ylim()
for phase in resonance_phases:
    ax_spec.axvline(phase * ISL / 2 / np.pi, linestyle="--")
ax_spec.set(xlabel="Phase", ylabel="Gain", title="Cavity spectrum")
ax_spec.grid()

# *** max gain and finesse ***
idx = np.where(spectrum > np.max(spectrum) * 0.5)
idx = idx[0]
fwhm = np.abs(freqs[idx[0]] - freqs[idx[-1]])
finesse = ISL / fwhm
gain = np.max(spectrum)
print('finesse = {:2.2f} | gain = {:2.2f}'.format(finesse,gain))


direct_for_figs = '/home/constance/Bureau/obsp'

# *** plot some resonant fields in the cavity ***
for num, phase in enumerate(resonance_phases):
    mode = get_total_field(fields, phase)
    phases_plots = np.linspace(-np.pi/50, np.pi/50, 3)
    for k, dp in enumerate(phases_plots):
        mode = get_total_field(fields, phase + dp)
        
        # *** plot 2d intensity profiles at resonance ***
        fig, ax_img = plt.subplots(dpi=100)
        title_fig = f'Mode_{num:.0f}_f0_{dp * ISL / 1e6:+.2f}MHz'.format(num, k)
        ax_img.set_title(title_fig, loc='right')
        im = ax_img.imshow((abs(mode) ** 2 + 1e-10), extent=(*(x[[0, -1]] * 1e3), *(y[[0, -1]] * 1e3)))
        phase_card = np.angle(mode)
        #im = ax_img.imshow((np.angle(mode) + 1e-10), extent=(*(x[[0, -1]] * 1e3), *(y[[0, -1]] * 1e3)))
        
        ax_img.set_ylim((-1.5*w*1e3,1.5*w*1e3))
        ax_img.set_xlim((-1.5*w*1e3,1.5*w*1e3))
        circle = mpatches.Circle(
            (0, 0), radius=w * 1e3,
            fill=False, edgecolor="#eeeeee",
            linestyle='--', linewidth=1.2,
            label=r"Power at $1/e^2$",
        )
        ax_img.add_patch(circle)
        ax_img.set(
            xlabel="x (mm)",
            ylabel="y (mm)",
        )
        ax_img.legend()
        #fig.savefig('2dprofile_r0_{}mm_desal_3015_sa3_only.svg', format = 'svg')
        
        #*** plot x-axis cut of the intensity at resonance ***
        fig_cut, ax_cut = plt.subplots()
        ax_cut.set_title(title_fig, loc='right')
        ax_cut.plot(x*1e3, (abs(mode) ** 2 + 1e-10)[N//2, :])
        if FD:
            ax_cut.plot(x*1e3, (abs(fermidirac(np.sqrt(X**2 + Y**2), 1.0, R0, 0, b).astype(np.complex64))**2 \
                                * np.max((abs(mode) ** 2 + 1e-10)))[N//2,:])
        else:
            ax_cut.plot(x*1e3, (abs(gaussian(X, Y, 1.0, w, x0=0, y0=0).astype(np.complex64))**2 \
                                * np.max((abs(mode) ** 2 + 1e-10)))[N//2,:])
        ax_cut.set_xlabel('x in mm')
        ax_cut.grid()
        #fig_cut.savefig('cut_r0_3mm_desal_3015_sa3_only.svg', format = 'svg')

        if FD:
            fig_err, ax_err = plt.subplots(1,1,sharex=True)
            I_comp = (np.abs(fermidirac(np.sqrt(X**2 + Y**2), 1.0, R0, 0, b).astype(np.complex64))**2) \
                      * np.max(abs(mode) ** 2)
            rel_err = np.abs(I_comp[N//2, :] - (abs(mode) ** 2 + 1e-10)[N//2, :]) / np.max(I_comp)
            ax_err.plot(x*1e3, rel_err)
            ax_err.grid()
            ax_err.set_xlabel('x in mm')
            ax_err.set_ylabel('Relative error to top-hat shape')  

plt.show()