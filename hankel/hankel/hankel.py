import scipy.special as ss
import numpy as np

def kernel(N):
    js = ss.jn_zeros(0, N+1)
    jN = js[N]
    j1s2 = ss.j1(js[:-1])**2

    Y = np.zeros((N, N))
    for m in range(N):
        Y[m, :] = 2 / (jN * j1s2) * ss.j0(js[m]/jN * js[:-1])

    return Y

def rp_space(N, R):
    js = ss.jn_zeros(0, N + 1)
    rs = js[:-1] * R / js[-1]
    ps = js[:-1] / R / (2*np.pi)
    
    return rs, ps

if __name__ == '__main__':
    N = 2**10

    Y = kernel(N)

    print(np.allclose(np.eye(N), np.matmul(Y, Y)))