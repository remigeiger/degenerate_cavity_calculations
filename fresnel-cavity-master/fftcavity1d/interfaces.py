import numpy as np

from . import cavity


class Interface(object):
    def process(self, efield):
        raise NotImplementedError

    def __add__(self, other):
        if isinstance(other, Interface):
            c = cavity.Cavity()
            c.append(self)
            c.append(other)
            return c
        if isinstance(other, cavity.Cavity):
            other.insert(0, self)
            return other
        raise TypeError('Expects Interface or Cavity')

    def __call__(self, efield):
        return self.process(efield)

    def __repr__(self):
        return '<cavity.Interface>'


class FreeSpace(Interface):
    R, T = 0.0, 1.0
    diameter = 1e10

    def __init__(self, distance):
        self.distance = distance

    def process(self, efield):
        return efield.propagated(self.distance)

    def __repr__(self):
        return 'FreeSpace(distance={s.distance})'.format(s=self)


class SphericalMirror(Interface):
    def __init__(self, R, radius, angle=0):
        self.R = R
        self.angle = angle
        self.T = 1-R
        self.radius = radius

    def process(self, efield):
        wf = efield.k * 0.5*self.radius * np.sqrt(1 + efield.x**2 / (0.5*self.radius)**2)
        return np.sqrt(self.R)*efield.dephased(-wf).tilted(2*self.angle)
        # return np.sqrt(self.R)*efield.lens(self.radius*0.5).tilted(2*self.angle)

    def tilt(self, angle):
        self.angle += angle

    def __repr__(self):
        return 'SphericalMirror(R={s.R}, radius={s.radius}, angle={s.angle})'.format(s=self)


class FlatMirror(Interface):
    def __init__(self, R, angle=0):
        self.R = R
        self.angle = angle
        self.T = 1-R

    def process(self, efield):
        return np.sqrt(self.R)*efield.tilted(2*self.angle)

    def tilt(self, angle):
        self.angle += angle

    def __repr__(self):
        return 'FlatMirror(R={s.R}, angle={s.angle})'.format(s=self)


class SphericalLens(Interface):
    def __init__(self, f, center=0, angle=0, T=1.0):
        self.f, self.T, self.R = f, T, 1-T
        self.angle = angle
        self.center = center

    def process(self, efield):
        wf = efield.k * self.f * np.sqrt(1 + (efield.x - self.center)**2 / self.f**2)
        return np.sqrt(self.T) * efield.dephased(-wf).tilted(self.angle)

    def tilt(self, angle):
        self.angle += angle

    def __repr__(self):
        return 'SphericalLens(f={s.f}, center={s.center}, angle={s.angle}, T={s.T})'.format(s=self)

class ParabolicLens(Interface):
    def __init__(self, f, center=0, angle=0, T=1.0):
        self.f, self.T, self.R = f, T, 1-T
        self.angle = angle
        self.center = center

    def process(self, efield):
        wf = efield.k * self.f * (1 + 0.5*(efield.x - self.center)**2 / self.f**2)
        return np.sqrt(self.T) * efield.dephased(-wf).tilted(self.angle)

    def tilt(self, angle):
        self.angle += angle

    def __repr__(self):
        return 'ParabolicLens(f={s.f}, center={s.center}, angle={s.angle}, T={s.T})'.format(s=self)

class PhaseMask(Interface):
    T = 1.0
    def __init__(self, phase_map):
        self.phase_map = phase_map

    def process(self, efield):
        return efield.dephased(self.phase_map)

    def __repr__(self):
        return '<cavity.PhaseMask>'

class Modulator(Interface):
    T = 1.0
    def __init__(self, modulation_map):
        self.modulation_map = modulation_map

    def process(self, efield):
        return self.modulation_map * efield

    def __repr__(self):
        return '<cavity.Modulator>'
