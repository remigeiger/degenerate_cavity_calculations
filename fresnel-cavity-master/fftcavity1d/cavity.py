import numpy as np

from . import interfaces


class Cavity(list):
    def __init__(self, cavity=None):
        super(Cavity, self).__init__()

    def append(self, interface):
        if not isinstance(interface, interfaces.Interface):
            raise ValueError('Cavity append expects Interface object.')
        super(Cavity, self).append(interface)

    def __add__(self, interface):
        self.append(interface)
        return self

    @property
    def interfaces(self):
        return list(self)

    @property
    def finess(self):
        r1, r2 = np.sqrt(self[0].R), np.sqrt(self[-1].R)
        T = np.prod(np.array([interface.T for interface in self[1:-1]]))
        return np.pi * np.sqrt(r1 * r2 * T**2) / (1 - r1 * r2 * T**2)

    def propagate_efield(self, E_in, N=1):
        if N is None:
            N = int(3 * self.finess)

        E = np.sqrt(1 - self[0].R) * E_in
        fields = np.zeros((N, E_in.E.size), dtype=np.complex128)
        fields[0, :] = E.E
        interfaces = (self[1:] + self[-2::-1])
        for i in range(1, N):
            for interface in interfaces:
                E = interface(E)
            fields[i, :] = E.E
        return fields
