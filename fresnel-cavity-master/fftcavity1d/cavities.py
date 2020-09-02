import numpy as np

from .cavity import Cavity
from . import interfaces


class MIGACavity(Cavity):
    def __init__(self, f, d1, d2, R1, R2, TL=1.0, parabolic_lens=True):
        super(MIGACavity, self).__init__()
        self.append(interfaces.FlatMirror(R=R1, angle=0))
        self.append(interfaces.FreeSpace(d1))
        if parabolic_lens:
            self.append(interfaces.ParabolicLens(f=f, center=0, angle=0, T=TL))
        else:
            self.append(interfaces.SphericalLens(f=f, center=0, angle=0, T=TL))
        self.append(interfaces.FreeSpace(d2))
        self.append(interfaces.FlatMirror(R=R2, angle=0))

        self._S2 = self[3]

    @property
    def M1(self):
        return self[0]

    @property
    def S1(self):
        return self[1]

    @property
    def L(self):
        return self[2]

    @property
    def S2(self):
        return self._S2

    @property
    def M2(self):
        return self[4]

    def set_d1(self, distance):
        self.S1.distance = distance

    def get_d1(self):
        return self.S1.distance

    d1 = property(get_d1, set_d1)

    def set_d2(self, distance):
        self.S2.distance = distance

    def get_d2(self):
        return self.S2.distance

    d2 = property(get_d2, set_d2)


class TwoMirrorCavity(Cavity):
    def __init__(self, R1, R2, L, radius1, radius2):
        self.radius1, self.radius2 = radius1, radius2
        super().__init__()
        if radius1 == np.inf:
            self.append(interfaces.FlatMirror(R=R1))
        else:
            self.append(interfaces.SphericalMirror(R=R1, radius=radius1))
        self.append(interfaces.FreeSpace(distance=L))
        if radius2 == np.inf:
            self.append(interfaces.FlatMirror(R=R2))
        else:
            self.append(interfaces.SphericalMirror(R=R2, radius=radius2))

    @property
    def M1(self):
        return self[0]

    @property
    def M2(self):
        return self[-1]

    @property
    def S(self):
        return self[1]

    def set_L(self, L):
        self.S.distance = L

    def get_L(self):
        return self.S.distance

    L = property(get_L, set_L)

    def waist(self, wavelength):
        L = self.S.distance
        g1 = 1.0 - L/self.radius1
        g2 = 1.0 - L/self.radius2
        if g1*g2 < 0 or 1 < g1*g2:
            return None
        return (np.sqrt(wavelength*L/np.pi)
                * (g1*g2*(1-g1*g2)/(g1+g2-2*g1*g2)**2)**0.25)
    
    def waist_position(self, wavelength):
        L = self.S.distance
        return L * (self.radius2 + L) / (self.radius2 + self.radius1 + 2*L)


class PlanePlaneCavity(Cavity):
    def __init__(self, R1, R2, L):
        super(PlanePlaneCavity, self).__init__()
        self.append(interfaces.FlatMirror(R=R1, angle=0))
        self.append(interfaces.FreeSpace(distance=L))
        self.append(interfaces.FlatMirror(R=R2, angle=0))

    @property
    def M1(self):
        return self[0]

    @property
    def M2(self):
        return self[-1]

    @property
    def S(self):
        return self[1]

    def set_L(self, L):
        self.S.distance = L

    def get_L(self):
        return self.S.distance

    L = property(get_L, set_L)
