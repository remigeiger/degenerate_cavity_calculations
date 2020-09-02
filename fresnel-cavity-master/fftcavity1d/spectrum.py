import numpy as np
import peakutils
import scipy.optimize as so
from tqdm import tqdm

from .functions import focused_linspace


class Spectrum(object):
    '''
    Spectrum created with a power function which should have resonances.
    It allows for simple resonance finding and refining.
    '''
    def __init__(self, power_fun):
        self.power = power_fun
        self.resonance_phases = None

    def compute_resonances(self, n_init=300, n_res=None, gain_thres=None, eps=0.02):
        '''
        Finds `n_res` highest resonances with a gain threshold `gain_thres` in the power_fun.
        If n_res is None, gets all resonances.
        If `gain_thres` is None, takes no condition on the resonances.
        Returns the resonance phases, stores them in self.resonance_phases.
        '''
        phases = np.linspace(0, 1.01*2 * np.pi, n_init)
        spectrum_log = np.log10(self(phases))

        m, M = spectrum_log.min(), spectrum_log.max()
        thres = (0.01 if gain_thres is None
                      else (np.log10(gain_thres) - m) / (M - m))
        spectrum_log = (spectrum_log - m) / (M - m)

        indexes = peakutils.indexes(spectrum_log,
                                    thres=thres, min_dist=5)
        indexes = sorted(indexes, key=lambda idx: spectrum_log[idx],
                         reverse=True)

        if n_res is not None:
            indexes = indexes[:n_res]

        resonance_phases = []
#        for p0 in phases[indexes]:
#            res = so.minimize(lambda phase: -self.power(float(phase)),
#                              method='L-BFGS-B',
#                              x0=p0,
#                              bounds=((p0 - 0.5 * eps * 2 * np.pi,
#                                       p0 + 0.5 * eps * 2 * np.pi),))
#            resonance_phases.append(float(res.x))
        
        dummy = phases[indexes]
        for num in tqdm(list(range(len(dummy)))):    
            res = so.minimize(lambda phase: -self.power(float(phase)),
                              method='L-BFGS-B',
                              x0=dummy[num],
                              bounds=((dummy[num] - 0.5 * eps * 2 * np.pi,
                                       dummy[num] + 0.5 * eps * 2 * np.pi),))
            resonance_phases.append(float(res.x))

        self.resonance_phases = resonance_phases
        return resonance_phases

    def phase_linspace(self, limits=(0, 2 * np.pi),
                       num_focused=100, num_unfocused=100,
                       focus_width=1/30):
        if self.resonance_phases is None:
            raise ValueError('Resonance phases have not been calculated.')
        else:
            if self.resonance_phases == []:
                p_begin, p_end = limits
            else:
                p_begin = min(limits[0], min(self.resonance_phases))
                p_end = max(limits[-1], max(self.resonance_phases))
            return focused_linspace(p_begin, p_end,
                                    self.resonance_phases, 2 * np.pi * focus_width,
                                    num_focused=num_focused,
                                    num_unfocused=num_unfocused)

    def __call__(self, phases=None):
        if phases is None:
            return self.spectrum(phases)
        else:
            return self.spectrum(phases)[1]

    def spectrum(self, phases=None, linspace_kwargs=None):
        if linspace_kwargs is None:
            linspace_kwargs = {}
        if phases is None:
            phases = self.phase_linspace(**linspace_kwargs)
        return phases, np.array([self.power(phase)
                                 for phase in phases])
