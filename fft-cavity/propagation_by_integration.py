import matplotlib.pyplot as plt
import numpy as np



def gaussian2d(X, Y, A, w, x0=0, y0=0):
    return A * np.exp(-((X-x0)**2 + (Y-y0)**2)/w**2)


n = 2**7
w = 1e-3
lda = 852e-9
a = 4*w
x = np.linspace(-a, a, n)
dx = 2*a/n
X, Y = np.meshgrid(x, x)

in_field = gaussian2d(X, Y, 1.0, w).astype(np.complex128)
in_field = in_field*0
in_field[np.sqrt(X**2 + Y**2) < 0.1e-3] = 1.0

z = 1e-3

out_field = in_field*0
for u in range(len(x)):
    for v in range(len(x)):
        # sum = 0
        # for k in range(len(x)):
        #     for l in range(len(x)):
        #         r = np.sqrt(z**2 + (x[u] - x[k])**2 + (x[v] - x[l])**2)
        #         sum += in_field[k, l] * np.exp(-1j*2*np.pi/lda*r)/r
        # out_field[u, v] = sum

        R = np.sqrt(z**2 + (X - x[u])**2 + (Y-x[v])**2)
        if v == 0 and u == n//2:
            fig, ax = plt.subplots()
            ax.imshow(R)
        integrand = in_field * np.exp(1j*2*np.pi/lda*R)/R
        out_field[u, v] = -1j/lda * np.trapz(x=x, y=np.trapz(x=x, y=integrand))



fig, ax = plt.subplots()
profile = abs(in_field[:, n//2])**2
ax.plot(x, profile / profile.max())
profile = abs(out_field[:, n//2])**2
ax.plot(x, profile / profile.max())

plt.show()