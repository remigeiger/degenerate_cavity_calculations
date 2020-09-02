import numpy as np
import scipy.special as ss

import matplotlib.pyplot as plt

import time

np.set_printoptions(suppress=True)

N = 2**12

js = ss.jn_zeros(0, N+1)
jN = js[N]
j1s2 = ss.j1(js[:-1])**2

t = time.time()
Y = np.zeros((N, N))
for m in range(N):
    Y[m, :] = 2 / (jN * j1s2) * ss.j0(js[m]/jN * js[:-1])
print(time.time() - t)

# t = time.time()
# print(np.allclose(np.dot(Y, Y), np.eye(N)))
# print(time.time() - t)

a = 7e-3

js = ss.jn_zeros(0, N)

rs = js * a / jN
ps = js / a / (2 * np.pi)

w = a/10
lda = 852e-9
zr = np.pi * w**2 / lda
E = np.exp(-rs**2/w**2)

d = 1

phase_shift = np.pi * lda * d * ps**2
t = time.time()
E_prop = Y.dot(Y.dot(E) * np.exp(-1j*phase_shift))
print(time.time() - t)

w_th = w * np.sqrt(1 + (d/zr)**2)
E_th = w / w_th * np.exp(-rs**2/w_th**2)
bt = Y.dot(E)

plt.figure()
plt.plot(ps, abs(bt))

plt.figure()
plt.plot(rs, abs(E)**2)
plt.plot(rs, abs(E_prop)**2)
plt.plot(rs, abs(E_th)**2, linestyle='--')

plt.show()