import numpy as np
import scipy.fftpack as sf
import matplotlib.pyplot as plt
import time

import efield


R1 = R2 = 0.9
Tl = 1 - 1e-3

nx = 2**14
w = 6e-3
xm = 10 * w
x = np.linspace(-xm, xm, nx)

lda = 852e-9

f = 0.2
d1 = d2 = f

freqs = sf.fftfreq(n=nx, d=x[1] - x[0])
lens_factor = np.exp(1j * 0.5 * x**2 / f**2)
phase_prop1 = 2 * np.pi / lda * d1 - np.pi * lda * d1 * freqs**2
phase_prop2 = 2 * np.pi / lda * d2 - np.pi * lda * d2 * freqs**2

factor1 = np.exp(-1j * phase_prop1)
factor2 = np.exp(-1j * phase_prop2)

p = np.sqrt(R1 * R2 * Tl)


def cavity(E):
    E = sf.ifft(sf.fft(E) * factor1)
    E *= lens_factor
    E = sf.ifft(sf.fft(E) * factor2)
    E = sf.ifft(sf.fft(E) * factor2)
    E *= lens_factor
    E = sf.ifft(sf.fft(E) * factor1)
    E *= p
    return E

def cavity2(E):
    freqs = sf.fftfreq(n=nx, d=x[1] - x[0])
    lens_factor = np.exp(1j * 0.5 * x**2 / f**2)
    phase_prop1 = 2 * np.pi / lda * d1 - np.pi * lda * d1 * freqs**2
    phase_prop2 = 2 * np.pi / lda * d2 - np.pi * lda * d2 * freqs**2
    factor1 = np.exp(-1j * phase_prop1)
    factor2 = np.exp(-1j * phase_prop2)
    E = sf.ifft(sf.fft(E) * factor1)
    E *= lens_factor
    E = sf.ifft(sf.fft(E) * factor2)
    E = sf.ifft(sf.fft(E) * factor2)
    E *= lens_factor
    E = sf.ifft(sf.fft(E) * factor1)
    E *= p
    return E


def cavity3(E):
    e = efield.EField(x=x, E=E, lda=lda)
    e = e.propagate(d1).lens(f).propagate(d2).propagate(d2).lens(f).propagate(d1)
    return e.E


E = np.exp(-x**2 / w**2, dtype=np.complex128)

t = time.time()
for n in range(1000):
    E = cavity(E)
print((time.time() - t) * 1e3)


E = np.exp(-x**2 / w**2, dtype=np.complex128)
t = time.time()
for n in range(1000):
    E = cavity2(E)
print((time.time() - t) * 1e3)


E = np.exp(-x**2 / w**2, dtype=np.complex128)
t = time.time()
for n in range(1000):
    E = cavity3(E)
print((time.time() - t) * 1e3)
