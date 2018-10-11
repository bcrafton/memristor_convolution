
import numpy as np
import pylab as plt
from keras.datasets import mnist

class Memristors:
    def __init__(self, N, M):
        self.U = 1e-16
        self.D = 10e-9
        self.W0 = 5e-9
        self.RON = 4e3
        self.ROFF = 10e3
        self.P = 5            
        self.W = np.ones(shape=(N, M)) * self.W0
        
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        
    def step(self, V, dt):
        self.R = self.RON * (self.W / self.D) + self.ROFF * (1 - (self.W / self.D))
        I = V / self.R
        F = 1 - (2 * (self.W / self.D) - 1) ** (2 * self.P)
        dwdt = ((self.U * self.RON * I) / self.D) * F
        dwdt += 0.1 * (self.W0 - self.W)
        self.W += dwdt * dt
        return I

#######################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.
img = np.pad(img, 2, mode='constant')

kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

#######################
   
steps = 1500
T = 20 * 1e-3
dt = T / steps
Ts = np.linspace(0, T, steps)

M = Memristors(28, 28)

#######################

Rs = []
Vs = []
Is = []

itrs = 3
fh = 3
fw = 3

for itr in range(itrs):
    for ii in range(fh):
        for jj in range(fw):
            for t in Ts:
                V = img[ii:28+ii, jj:28+jj] * kernel[ii][jj]
                # V = img[ii:28+ii, jj:28+jj] - kernel[ii][jj]
                I = M.step(V, dt)
                Rs.append(M.R[11, 14])

'''
total_T = T*itrs*fh*fw
total_steps = steps*itrs*fh*fw
Ts = np.linspace(0, total_T, total_steps)
plt.plot(Ts, Rs)
plt.show()
'''

'''
print (M.R)
print (np.max(M.R), np.min(M.R))
'''

R = M.R / np.max(M.R)
plt.imshow(R, cmap=plt.cm.gray)
plt.show()

