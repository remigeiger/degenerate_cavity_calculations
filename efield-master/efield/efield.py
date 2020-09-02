from __future__ import division

import numpy as np
import scipy.fftpack as sf
import copy

import matplotlib.pyplot as plt


class EField(object):
    # required for right multiplication by numpy array
    __array_priority__ = 10000

    def __init__(self, x, E, lda=852e-9, *,
                 normalize_power=False, radial=False,
                 prop_type='fresnel'):
        if len(x) != len(E):
            raise ValueError('x and E should have the same size.')

        self.x = x
        self.E = E.astype(dtype=np.complex128)
        self.lda = lda
        self.radial = radial
        self.prop_type = prop_type

        self.freqs = sf.fftfreq(len(x), x[1] - x[0])

        if normalize_power:
            self.E /= np.sqrt(self.P)

    def __len__(self):
        return len(self.x)

    @property
    def I(self):
        return np.abs(self.E)**2

    @property
    def phase(self):
        return np.angle(self.E)

    @property
    def k(self):
        return 2 * np.pi / self.lda

    @property
    def P(self):
        if self.radial:
            return np.trapz(x=self.x, y=np.pi * abs(self.x) * self.I)
        return np.trapz(x=self.x, y=self.I)

    def propagated(self, dist):
        freqs = self.freqs
        if self.prop_type == 'fresnel':
            # full paraxial
            phase_shift = self.k * dist - np.pi * self.lda * dist * freqs**2
            tfE = sf.fft(self.E)
            E_out = sf.ifft(tfE * np.exp(-1j * phase_shift))

        elif self.prop_type == 'evanescent':
            # square root with evanescent
            phase_shift = self.k * dist * np.sqrt(1 - self.lda**2 * freqs**2 + 0j)
            evanescent_idx = 1 - self.lda**2 * freqs**2 < 0        
            phase_shift[evanescent_idx] = - phase_shift[evanescent_idx]
            tfE = sf.fft(self.E)
            E_out = sf.ifft(tfE * np.exp(-1j * phase_shift))
        else:
            raise ValueError('{} is not a valid propagation type'.format(self.prop_type))

        out = EField.from_EField(self)
        out.E = E_out
        return out


    def dephased(self, phase):
        out = EField.from_EField(self)
        out.E = out.E * np.exp(-1j*phase)
        return out

    def attenuated(self, attenuation):
        out = EField.from_EField(self)
        out.E = out.E * attenuation
        return out

    def tilted(self, angle):
        if angle == 0:
            phase = 0
        else:
            phase = self.k * angle * self.x
        return self.dephased(phase)

    def propagate(self, dist):
        self = self.propagated(dist)
    
    def dephase(self, phase):
        self = self.dephased(phase)

    def attenuate(self, attenuation):
        self = self.attenuated(attenuation)
    
    def tilt(self, angle):
        self = self.tilted(angle)

    def __repr__(self):
        return '<EField lda={lda} points={points}>'.format(lda=self.lda,
                                                           points=self.x.size)

    def plot(self, fig=None, phase=True, intensity=True,
             normalize=False,
             *args, **kwargs):
        """ plots the Intensity and the phase on two subplots"""
        if fig is None:
            fig, (axI, axP) = self.create_figure()
        elif len(fig.axes) == 2:
            axI = fig.axes[0]
            axP = fig.axes[1]
        else:
            raise ValueError('fig argument should have two axis')

        if intensity:
            norm = 1.0
            if normalize:
                norm = self.I.max()
            axI.plot(self.x * 1e3, self.I / norm, *args, **kwargs)
        else:
            axI.plot([], [], *args, **kwargs)
        if phase:
            # phase = -np.unwrap(self.phase - self.phase[self.phase.size // 2])
            phase = -np.unwrap(self.phase)
            axP.plot(self.x * 1e3, (phase - phase[phase.size // 2]) / np.pi,
                     *args, **kwargs)
        else:
            axP.plot([], [], *args, **kwargs)

        return fig, (axI, axP)

    @classmethod
    def from_EField(cls, E):
        """Initializer from other EField"""
        if not isinstance(E, EField):
            raise ValueError('EField.from_EField expects EField as argument.')
        return copy.deepcopy(E)

    def __add__(self, rhs):
        if not np.allclose(self.x, rhs.x):
            raise ValueError('Both fields should be defined '
                             'on the same interval.')
        return EField(copy.copy(self.x), self.E + rhs.E)

    def __mul__(self, rhs):
        E_out = EField.from_EField(self)
        E_out.E = E_out.E * rhs
        return E_out

    __rmul__ = __mul__

    @classmethod
    def create_figure(cls):
        fig, (axI, axP) = plt.subplots(2, 1, sharex=True)
        axI.grid()
        axP.grid()
        axP.set_xlabel('x (mm)')
        axP.set_ylabel('Phase ($\pi rad$)')
        axI.set_ylabel(r'Intensity ($W/m^2$)')
        return fig, (axI, axP)
