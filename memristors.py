
import numpy as np
import pylab as plt

class Memristors:
    
    def __init__(self, N, M):
        self.U = 1e-16
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 5e4
        self.ROFF = 1e6
        self.P = 5            
        self.W = np.ones(shape=(N, M)) * self.W0
        
    def step(self, V, dt):
        R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = V / R
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        self.W += dwdt * dt
        
        return I
        
steps = 1500
T = 2 * np.pi
dt = T / steps

M = Memristors(10, 10)

Vs = []
Is = []

for t in range(steps): 
    V = np.sin(t * dt) * np.ones(shape=(10, 10))
    I = M.step(V, dt)
    
    Vs.append( V[0, 0] )
    Is.append( I[0, 0] )
    
plt.plot(Vs, Is)
plt.show()
