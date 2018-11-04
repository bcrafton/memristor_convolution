
import numpy as np
import pylab as plt

U = 1e-16
D = 10e-9
W0 = 5e-9
RON = 4e3
ROFF = 10e3
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
    
    if t < np.pi / 2.:
        V = 1 * np.sin(t)
    else:
        V = 0.
    
    R = RON * (W / D) + ROFF * (1 - (W / D))
    I = V / R
    
    F = 1 - (2 * (W / D) - 1) ** (2 * P)
    dwdt = ((U * RON * I) / D) * F
    dwdt += 2. * (W0 - W)
    W += dwdt * dt
    
    Is.append(I)
    Vs.append(V)
    Rs.append(R)
    
    print ("V", V, "I", I, "R", R, "W", W, "F", F, "dw", dwdt * dt)
    
fig = plt.figure(figsize=(15, 15))

plt.subplot(3, 1, 1)
plt.plot(Ts, Vs)
plt.xlabel('Time', fontdict={'size':18})
plt.ylabel('Voltage', fontdict={'size':18})
    
plt.subplot(3, 1, 2)
plt.plot(Ts, Rs)
plt.xlabel('Time', fontdict={'size':18})
plt.ylabel('Resistance', fontdict={'size':18})

plt.subplot(3, 1, 3)
plt.plot(Vs, Is)
plt.xlabel('Voltage', fontdict={'size':18})
plt.ylabel('Current', fontdict={'size':18})

plt.savefig('memristor_plots.png')
