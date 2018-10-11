
import numpy as np
import pylab as plt

U = 1e-16
D = 10e-9
W0 = 5e-9
RON = 5e4
ROFF = 1e6
P = 5

steps = 1500
T = 2 * np.pi
dt = T / steps
Ts = np.linspace(0, T, steps)

W = W0

Is = []
Vs = []
Rs = []

for t in Ts:
    
    if t < np.pi:
        V = 1 * np.sin(t)
    else:
        V = 0.
    
    R = RON * (W / D) + ROFF * (1 - (W / D))
    I = V / R
    
    F = 1 - (2 * (W / D) - 1) ** (2 * P)
    dwdt = ((U * RON * I) / D) * F
    dwdt += W0 - W
    W += dwdt * dt
    
    Is.append(I)
    Vs.append(V)
    Rs.append(R)
    
    print ("V", V, "I", I, "R", R, "W", W, "F", F, "dw", dwdt * dt)
    
plt.plot(Ts, Rs)
# plt.plot(Ts, Vs)

print (np.max(Vs), np.min(Vs))

plt.show()
