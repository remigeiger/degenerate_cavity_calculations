import numpy as np
import numba
import matplotlib.pyplot as plt
import logging
import time
import cmath

from efield import EField
from . import spectrum
from .functions import round_to_odd


class HandlerError(Exception): pass
class NotReady(HandlerError): pass


class CavityHandler(object):
    def __init__(self, cavity, radial=False):
        self.cavity = cavity
        self.fields = None
        self.input_efield = None
        self.spectrum = None
        self.radial = radial
        self.logger = logging.getLogger(__name__)

    def fields_are_calculated(self):
        return self.fields is not None

    def calculate_fields(self, input_efield, N=None):
        '''
        Gets the fields propagated inside the cavity at every round trip.
        '''
        self.input_efield = EField.from_EField(input_efield)
        self.fields = self.cavity.propagate_efield(input_efield, N=N)

    def power(self, phase):
        '''
        Calculates the power of the beam inside the cavity at the
        given phase shift.
        '''
        intensity = abs(self.compute_total_field(phase))**2
        if self.radial:
            return np.trapz(y=np.pi * abs(self.input_efield.x) * intensity,
                            x=self.input_efield.x)
        else:
            return np.trapz(y=intensity,
                            x=self.input_efield.x)

    @numba.jit
    def compute_total_field(self, phase):
        '''
        Computes the total field at a given phase. It sums all the calculated
        fields at every round trip with a phase shift given by `phase`.
        '''
        if not self.fields_are_calculated():
            self.logger.error('Fields are not calculated')
            message = 'Calculate fields before computing total field'
            raise NotReady(message)
        fields = self.fields
        shape = fields.shape
        total_field = np.zeros(shape[1], dtype=np.complex128)
        for i in range(shape[0]):
            factor = np.exp(-1j * (i + 1) * phase)
            for k in range(shape[1]):
                total_field[k] = total_field[k] + fields[i, k] * factor
        return total_field

    def compute_resonances(self, n_init=300, n_res=None, gain_thres=None, **kwargs):
        '''
        Creates a Spectrum instance and finds the resonances positions.
        Returns the resonance phases and stores a Spectrum instance.
        '''
        self.spectrum = spectrum.Spectrum(power_fun=self.power)
        t = time.time()
        self.spectrum.compute_resonances(n_init, n_res, gain_thres, **kwargs)
        message = 'Time to compute resonances : {0:.3f} s'
        self.logger.debug(message.format(time.time() - t))

        return self.spectrum.resonance_phases

    def efield_at(self, phase):
        '''
        Returns the EField circulating in the cavity at given phase
        '''
        E_out = EField.from_EField(self.input_efield)
        E_out.E = self.compute_total_field(phase)
        return E_out

    def plot_spectrum(self, phases=None, ax=None, show_resonances=False,
                      *args, **kwargs):
        '''
        Plots the spectrum of the cavity at the defined phases or with a
        focused linspace defined by the resonances.
        If show_resonances is true, will put vertical lines at resonances.
        '''
        if ax is None:
            fig, ax = plt.subplots()
        phases, spectrum = self.spectrum(phases)
        ax.semilogy(phases, spectrum, *args, **kwargs)
        if show_resonances:
            yl = ax.get_ylim()
            for num, phase in enumerate(self.spectrum.resonance_phases):
                ax.plot([phase] * 2, yl, linestyle='--',
                        label='{} : {:5.3f} rad'.format(num + 1, phase))
            ax.legend()
        
        return ax

    def plot_fields(self, fig=None):
        '''
        Plots the fields at 7 round trips distributed through
        all the round trips.
        '''
        if not self.fields_are_calculated():
            self.logger.error('Fields are not calculated')
            message = 'Calculate fields before ploting'
            raise NotReady(message)
        if fig is None:
            fig, (axI, axP) = EField.create_figure()

        step = round_to_odd(self.fields.shape[0] / 7)
        for i in range(0, self.fields.shape[0], step):
            efield = EField(self.input_efield.x, self.fields[i, :])
            fig, (axI, axP) = efield.plot(fig, label='# {}'.format(i),
                                          normalize=True)
        axI.legend()
        return fig, (axI, axP)
