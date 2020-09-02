import numpy as np
import matplotlib.pyplot as plt


def piston(u):
    out = np.zeros(u.shape)
    u = abs(u)
    u_ok = abs(u) <= 1
    out[u_ok] = (0 * u[u_ok] + 1)/np.sqrt(np.pi)
    return out

def spherical_2(u):
    out = np.zeros(u.shape)
    u = abs(u)
    u_ok = abs(u) <= 1
    out[u_ok] = np.sqrt(3/np.pi) * (2*u[u_ok]**2 - 1)
    return out

def spherical_3(u):
    out = np.zeros(u.shape)
    u = abs(u)
    u_ok = abs(u) <= 1
    out[u_ok] = np.sqrt(5/np.pi) * (6*u[u_ok]**4 - 6*u[u_ok]**2 + 1)
    return out

def spherical_4(u):
    out = np.zeros(u.shape)
    u = abs(u)
    u_ok = abs(u) <= 1
    out[u_ok] = 1/np.sqrt(0.448799)*(20*u[u_ok]**6 - 30*u[u_ok]**4 + 12*u[u_ok]**2 - 1)
    return out

def decompose_spherical(u, aberrated):
    coeffs = [np.trapz(x=u, y=2*np.pi*u*fun(u)*aberrated) for fun in [piston, spherical_2, spherical_3, spherical_4]]
    return coeffs

def recompose_spherical(u, coeffs):
    return sum([coeffs[i] * fun(u) for i, fun in enumerate([piston, spherical_2, spherical_3, spherical_4])])


if __name__ == '__main__':
    R = 100e-3
    f = 2*R

    lda = 852e-9
    k = 2*np.pi / lda

    aperture = 50.8e-3
    nx = 2**14
    r = np.linspace(0, aperture/2, nx)
    u = r / (aperture/2)

    wf_full = k*f * np.sqrt(1 + r ** 2 / f ** 2) - k*f
    wf_paraxial = k*f * (1 + 0.5 * r ** 2 / f ** 2) - k*f

    coeffs_paraxial = decompose_spherical(u, wf_paraxial)
    coeffs_full = decompose_spherical(u, wf_full)


    fig, axes = plt.subplots(3, 1, sharex=True)
    axes[0].plot(u, wf_full)
    axes[0].plot(u, wf_paraxial)

    axes[1].plot(u, wf_full - wf_paraxial)

    # axes[2].plot(u, coeffs_full[0]*piston(u) + coeffs_full[1]*spherical_2(u) + coeffs_full[2]*spherical_3(u) + coeffs_full[3]*spherical_4(u))
    axes[2].plot(u, (coeffs_full[2]*spherical_3(u) + coeffs_full[3]*spherical_4(u)) / (2*np.pi))

    print(coeffs_paraxial)
    print(coeffs_full)


    plt.show()
