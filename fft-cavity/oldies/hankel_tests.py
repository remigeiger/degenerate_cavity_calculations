import numpy as np

def fast_ht(f):
    n = length(f)
    dt = 1/n
    K = 9

    hk = np.array([0.318, 0.19, 0.35, 0.82, 1.8, 3.9, 8.3, 19.6, 48.3])
    ldak = np.array([0, -2.1, -6.2, -22.4, -92.5, -414.5, -1889.4, -8990.9, -47391.1])

    C = np.ones(K)
    s = 1-1e-9
    
    def beta0(n, lda):
        return 2*(n-s) / (lda + s) . (lda + 2) * ((n - s) + (lda - n + 2)*(n/(n - s))**(lda + s))
    
    def beta1(n, lda):
        return -2*(n-s) / (lda + s) . (lda + 2) * ((lda + n + s) - n*(n/(n - s))**(lda + s))
    
    B0 = lambda n: (hk * beta0(n, ldak))
    B1 = lambda n: (hk * beta1(n, ldak))
    Phi = lambda n: np.diag(n / (n-s)) ** ldak

    x = np.zeros(k)
    g = np.zeros(n)

    fp = np.array([f, 0])
    for n