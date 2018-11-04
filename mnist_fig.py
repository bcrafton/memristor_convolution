
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from PIL import Image

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.
plt.imsave("mnist.png", img, cmap=plt.cm.gray) 

img = Image.open('mnist.png')
basewidth = 300
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('mnist.png') 

####################################################################

fh = 3
fw = 3

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.
img = np.pad(img, 2, mode='constant')

for ii in range(fh):
    for jj in range(fw):
        img[11+ii, 14+jj] = -1.

plt.imsave("mnist_highlight.png", img, cmap=plt.cm.gray) 

img = Image.open('mnist_highlight.png')
basewidth = 300
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('mnist_highlight.png') 

####################################################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.
img = np.pad(img, 2, mode='constant')

kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

conv_img = np.zeros(shape=(28, 28))
for ii in range(3):
    for jj in range(3):
        conv_img += kernel[ii][jj] * img[ii:28+ii, jj:28+jj]

plt.imsave("mnist_conv.png", conv_img, cmap=plt.cm.gray) 

img = Image.open('mnist_conv.png')
basewidth = 300
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('mnist_conv.png') 

####################################################################

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

itrs = 3
fh = 3
fw = 3

#######################

for itr in range(itrs):
    for ii in range(fh):
        for jj in range(fw):
            for t in Ts:
                V = img[ii:28+ii, jj:28+jj] - kernel[ii][jj]
                I = M.step(V, dt)

R = M.R / np.max(M.R)
plt.imsave("memristor_conv_bad.png", R, cmap=plt.cm.gray) 

img = Image.open('memristor_conv_bad.png')
basewidth = 300
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('memristor_conv_bad.png') 

####################################################################

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

itrs = 3
fh = 3
fw = 3

#######################

for itr in range(itrs):
    for ii in range(fh):
        for jj in range(fw):
            for t in Ts:
                V = img[ii:28+ii, jj:28+jj] * kernel[ii][jj]
                I = M.step(V, dt)

R = M.R / np.max(M.R)
plt.imsave("memristor_conv_good.png", R, cmap=plt.cm.gray) 

img = Image.open('memristor_conv_good.png')
basewidth = 300
wpercent = (basewidth/float(img.size[0]))
hsize = int((float(img.size[1])*float(wpercent)))
img = img.resize((basewidth,hsize), Image.ANTIALIAS)
img.save('memristor_conv_good.png') 

####################################################################

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

itrs = 1
fh = 3
fw = 3

for itr in range(itrs):
    for ii in range(fh):
        for jj in range(fw):
            for t in Ts:
                V = img[ii:28+ii, jj:28+jj] - kernel[ii][jj]
                I = M.step(V, dt)
                Rs.append(M.R[11, 14])

total_T = T*itrs*fh*fw
total_steps = steps*itrs*fh*fw
Ts = np.linspace(0, total_T, total_steps)
plt.plot(Ts, Rs)
plt.savefig('memristor_plots_bad.png')
plt.clf()

####################################################################

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

itrs = 1
fh = 3
fw = 3

for itr in range(itrs):
    for ii in range(fh):
        for jj in range(fw):
            for t in Ts:
                V = img[ii:28+ii, jj:28+jj] * kernel[ii][jj]
                I = M.step(V, dt)
                Rs.append(M.R[11, 14])

total_T = T*itrs*fh*fw
total_steps = steps*itrs*fh*fw
Ts = np.linspace(0, total_T, total_steps)
plt.plot(Ts, Rs)
plt.savefig('memristor_plots_good.png')
plt.clf()

####################################################################

