import numpy as np
from efield import EField
import matplotlib.pyplot as plt
from Cavity_toolbox import compute_power, compute_fields, compute_total_field
import scipy.optimize as so
from test_cavity_fft import round_to_odd


def gaussian(x, amplitude, center, waist, offset):
    return amplitude * np.exp(-(x-center)**2/waist**2) + offset


win = 5e-3
N = 2**14
x0 = 50*win
x = np.linspace(-x0, x0, N)

lda = 770e-9
E_ref = EField(x, E=gaussian(x, amplitude=1, center=0*win,
                             waist=win, offset=0),
               lda=lda, normalize=True)

angle_beam = 0e-3
angle_m1 = 0e-6
angle_m2 = 0e-3

r1 = np.sqrt(0.99)
r2 = np.sqrt(0.99)
L = 1e-3


def cavity_fun(efield):
    return efield.propagate(L).tilt(2*angle_m2).propagate(L).tilt(2*angle_m1)


F = np.pi*np.sqrt(r1*r2)/(1-r1*r2)
print('Finesse : {:.0f}'.format(F))
N_rt = int(2*F)

input_efield = np.sqrt(1-r1**2)*EField.from_EField(E_ref).tilt(angle_beam)

fields = compute_fields(cavity_fun, input_efield, r1*r2, N_rt)
res = so.minimize_scalar(lambda dL: -compute_power(x, fields, lda, dL),
                         bounds=(0, lda/2), method='bounded',
                         options={'xatol': lda/10000})

n = 200
DdL = lda/2/20
dLs = np.linspace(0, 1.2*lda/2, n)
dLs = np.hstack([np.linspace(res.x - DdL, res.x + DdL, n),
                 np.linspace(res.x + DdL, res.x + lda/2 - DdL, 15)])
power = [compute_power(x, fields, lda, dL)
         for dL in dLs]

E_cav = EField(x, compute_total_field(fields, lda, res.x))
print('Maxima found at dL = {:.3f} lda'.format(res.x/lda))

fig, (axI, axP) = E_ref.plot(label='Input', normalize=True)
for i in range(0, N_rt, round_to_odd(N_rt/5)):
    EField(x, fields[i, :]).plot(fig, label='# {}'.format(i),
                                 normalize=True)
axI.legend(loc='upper right')

plt.figure()
plt.semilogy(dLs/lda, power)
plt.xlabel(r'$\Delta L$ ($\lambda$)')
plt.plot([res.x/lda, res.x/lda], plt.ylim(), '--')
plt.ylabel('Power')

fig, (axI, axP) = E_ref.plot(label='Input', normalize=True)
E_cav.plot(fig, label='Mode', normalize=True)

plt.show()
