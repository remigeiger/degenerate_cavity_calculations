import matplotlib.pyplot as plt
import numpy as np

from estimate_finesse import estimate_finesse

def compute_finesse(R1, R2, TS):
    B = 2*TS*np.sqrt(R1*R2)/(1+TS**2*R1*R2)
    alpha = TS*np.sqrt(R1*R2)
    M = 4*alpha / (1-alpha)**2
    print(np.sqrt(1/M))
    finess1 = 2*np.pi / (2*np.arccos(2 - 1/B))
    finess2 = 2*np.pi / (4*np.arcsin(np.sqrt(1/M)))
    print(finess1, finess2)
    print(2-1/B)
    return finess2, np.arccos(2 - 1/B)

def compute_finesse2(R1, R2, TS):
    return np.pi * np.sqrt(np.sqrt(R1*R2) * TS) / (1 - np.sqrt(R1*R2) * TS)

phase = np.linspace(-0.1*np.pi, 0.1*np.pi, 10000)
R1 = 0.985
R2 = 0.985
TS = 0.99

amp_gain = np.sqrt(1 - R1) / (1 - TS * np.sqrt(R1*R2) * np.exp(1j*phase))
power_gain = (1-R1) / (1 + TS**2*R1*R2 - 2*TS*np.sqrt(R1*R2)*np.cos(phase))

finesse = estimate_finesse(phase, power_gain)
print('Estimated finesse : ', finesse)
computed_finesse, phi_0 = compute_finesse(R1, R2, TS)
print('Computed finesse 1 : ', computed_finesse)
computed_finesse2 = compute_finesse2(R1, R2, TS)
print('Computed finesse 2 : ', computed_finesse2)

G_th = (1-R1) / (1 - TS*np.sqrt(R1*R2)*(2 - TS*np.sqrt(R1*R2)))
print('Computed gain : ', G_th)
G_est = np.max(power_gain)
print('Estimated gain : ', G_est)

fig, ax = plt.subplots()
ax.plot(phase, power_gain)
ylim = ax.get_ylim()
ax.set_ylim(0, ylim[1])
ylim = ax.get_ylim()
ax.plot([phi_0]*2, ylim, linestyle='--')




plt.show()