import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as nf
from tqdm import tqdm

from fftcavity1d.spectrum import Spectrum
from zernike import zernike_combination


def gaussian(X, Y, amplitude, waist, x0=0, y0=0):
    return amplitude * np.exp(-((X-x0)**2 + (Y-y0)**2)/waist**2)

def get_prop_phase_shift(dist, lda, Freqs):
    return 2*np.pi/lda * dist - np.pi * lda * dist * (Freqs[0]**2 + Freqs[1]**2)


def get_lens_wf(f, lda, X, Y):
    # wf = 2*np.pi/lda * f * np.sqrt(1 + (X**2 + Y**2) / f ** 2)
    wf = 2*np.pi/lda * f * (1 + 0.5 * (X**2 + Y**2) / f ** 2)
    return wf

def propagate(field, prop_factor):
    return nf.ifft2(nf.fft2(field) * prop_factor)

def get_fields(field, cavity_fun, n_rt):
    fields = np.empty((*field.shape, n_rt), np.complex64)
    fields[:, :, 0] = field
    for num in tqdm(range(n_rt - 1)):
        fields[:, :, num + 1] = cavity_fun(fields[:, :, num])
    return fields

def get_total_field(fields, phase):
    phases = np.exp(1j*np.arange(fields.shape[2])*phase)
    return np.einsum('ijk,k->ij', fields, phases)
    # return (fields * phases[:, np.newaxis, np.newaxis]).sum(axis=2)

#def power(field):
#    return np.sum(abs(field)**2)
def power(field):
    return np.trapz(np.trapz(abs(field)**2))

def power_at_phase(fields, phase):
    total_field = get_total_field(fields, phase)
    return power(total_field)



N = 2**8
w = 2.0e-3
lda = 852e-9
k0 = 2*np.pi/lda
a = 3*w
f = 0.2

#print('dx = {} um'.format())

x = np.linspace(-a, a, N)
y = np.linspace(-a, a, N)
X, Y = np.meshgrid(x, y)
freqsx = nf.fftfreq(N, x[1] - x[0])
freqsy = nf.fftfreq(N, y[1] - y[0])
Freqs = np.meshgrid(freqsx, freqsy)

y = None
freqsx = None
freqsy = None


r_optic = 25e-3
orders = [(2, 0), (2, +2), (2, -2), (3, 1), (4, 0)]
coeffs = [0, 20, 0.0, 0.0, 00]
phi = np.arctan2(X, Y)

aberration_map = zernike_combination(orders, coeffs, np.sqrt(X**2 + Y**2)/r_optic, phi) #- 0.01*lda*Y/a

# fig, ax = plt.subplots()
# mappable = ax.imshow(aberration_map)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, aberration_map)
# # plt.colorbar(mappable)
# plt.show()
# exit()

fig, (ax_x, ax_y) = plt.subplots(2, 1)
ax_x.plot(aberration_map[N//2, :])
ax_y.plot(aberration_map[:, N//2])



# in_field = gaussian(X, Y, 1.0, w).astype(np.complex64)
# in_field /= np.sqrt(power(in_field))
#
# fig, (ax1, ax2) = plt.subplots(2, 1)
# ax1.plot(x*1e6, abs(in_field[:, N//2])**2, label='1')
#
# prop_phase_shift = get_prop_phase_shift(f, lda, Freqs)
# prop_factor = np.exp(-1j*prop_phase_shift)
#
# lens_wf = 2 * np.pi / lda * f * np.sqrt(1 + (X ** 2 + Y ** 2) / f ** 2)
# lens_factor = np.exp(1j * lens_wf)
#
# out_field = propagate(in_field, prop_factor)
# ax1.plot(x*1e6, abs(out_field[:, N//2])**2, label='2')
# out_field *= lens_factor
# out_field = propagate(out_field, prop_factor)
# # out_field[abs(x) > 500e-6] = 0
# ax2.plot(x*1e6, abs(out_field[:, N//2])**2, label='3')
# out_field = propagate(out_field, prop_factor)
# ax1.plot(x*1e6, abs(out_field[:, N//2])**2, label='4')
# out_field *= lens_factor
# out_field = propagate(out_field, prop_factor)
# ax1.plot(x*1e6, abs(out_field[:, N//2])**2, label='5')
#
# print(out_field.nbytes/1e6)
#
# ax1.legend()
# ax2.legend()
#
# plt.show()
#
#
# exit()


def MIGA_cavity_fun(d1, d2, f, R, TL):
    lens_wf = 2*np.pi/lda * f * (1 + 0.5 * (X**2/f**2 + Y**2/f**2) )
    # lens_wf = 2*np.pi/lda * f * np.sqrt(1 + (X**2 + Y**2) / f**2)
    lens_factor = np.exp(1j*lens_wf)
    prop1_phase_shift = get_prop_phase_shift(d1, lda, Freqs)
    prop1_factor = np.exp(-1j*prop1_phase_shift)
    prop2_phase_shift = get_prop_phase_shift(d2, lda, Freqs)
    prop2_factor = np.exp(-1j*prop2_phase_shift)
    aberr_factor = np.exp(1j*2*np.pi*aberration_map)
    def cavity_fun(in_field):
        out = propagate(in_field, prop1_factor)
        out *= lens_factor
        out = propagate(out, prop2_factor)
        out = propagate(out, prop2_factor)
        out *= lens_factor
        out = propagate(out, prop1_factor)
        out *= aberr_factor
        out *= aberr_factor

        return out * R * TL
    return cavity_fun

# def flat_concave_cavity(L, R):
#     mirror_wavefront = 2*np.pi/lda * 0.5*R * (1 + 0.5 * (X**2/(0.5*R)**2 + Y**2/(0.5*R)**2) )
#     mirror_factor = np.exp(1j*mirror_wavefront)
#     prop_phase_shift = get_prop_phase_shift(L, lda, Freqs)
#     prop_factor = np.exp(-1j*prop_phase_shift())



R = 0.9
TL = 1.0
d1 = f
d2 = f

cavity_fun = MIGA_cavity_fun(d1, d2, f, R, TL)

in_field = gaussian(X, Y, 1.0, w, 0, 0).astype(np.complex64)
in_field /= np.sqrt(power(in_field))



# plt.show()
#
# exit()

print('Computing fields.')
fields = get_fields(np.sqrt(1-R)*in_field, cavity_fun, 50)
print('Fields computed.')
print('Memory used : {:.0f} Mb'.format(fields.nbytes / 1e6))

# total_field = get_total_field(fields, 0)

s = Spectrum(lambda phase: power_at_phase(fields, phase))

print('Computing resonances.')
resonance_phases = s.compute_resonances(n_res=1, gain_thres=0.3)
print('Resonances computed.')

phases, spectrum = s.spectrum()


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

# fig, (axI, axP) = plt.subplots(2, 1, sharex=True)
# for num in range(0, fields.shape[2], 2*(fields.shape[2]//10)+1):
#     plot_field(fields[:, :, num], axis=1, axI=axI, axP=axP, label='{}'.format(num))
# axI.legend()
# axP.set_ylim(-0.2, 0.2)


fig, ax = plt.subplots()
ax.plot(phases, spectrum, color='k')
yl = ax.get_ylim()
for phase in resonance_phases:
    ax.plot([phase]*2, yl, linestyle='--')


for num, phase in enumerate(resonance_phases):
    mode = get_total_field(fields, phase)
    fig, (ax_img, ax_profile) = plt.subplots(1, 2)
    fig.canvas.set_window_title('Mode {:.0f}'.format(num))
    ax_img.imshow((abs(mode)**2+1e-10))
    # plot_field(mode, axI=ax_profile)

    for k, dp in enumerate(np.linspace(-np.pi/100, np.pi/100, 3)):

        mode = get_total_field(fields, phase+dp)
        fig, (ax_img, ax_profile) = plt.subplots(1, 2)
        fig.canvas.set_window_title('Mode {:.0f} + {:f}'.format(num, k))
        ax_img.imshow((abs(mode)**2+1e-10))
        # plot_field(mode, axI=ax_profile)

plt.show()



