import numpy as np
from tqdm import tqdm
import cmath
import numba


def compute_fields(cavity_fun, input_efield, losses_per_rt, N_rt):
    E = input_efield
    fields = np.zeros((N_rt, len(E)), dtype=complex)
    C = 1
    for i in range(N_rt):
        E = cavity_fun(E)
        C *= losses_per_rt
        fields[i, :] = C*E.E
    return fields


@numba.jit
def compute_total_field(fields, lda, dL):
    shape = fields.shape
    total_field = np.zeros(shape[1], dtype=complex)
    for i in range(shape[0]):
        factor = cmath.exp(-1j*2*np.pi/lda*(i+1)*2*dL)
        for k in range(shape[1]):
            total_field[k] = total_field[k] + fields[i, k] * factor
    return total_field

# def compute_total_field(fields, lda, dL):
#     rt = np.arange(1, fields.shape[0]+1)
#     dephasing = np.exp(-1j*2*pi/lda*rt*2*dL)
#     return np.einsum('i,ij->j', dephasing, fields)


def compute_power(x, fields, lda, dL):
    return np.trapz(np.pi*abs(x)*abs(compute_total_field(fields, lda, dL))**2, x)
