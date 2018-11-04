
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist

def convolve2d(image, kernel):
    # This function which takes an image and a kernel 
    # and returns the convolution of them
    # Args:
    #   image: a numpy array of size [image_height, image_width].
    #   kernel: a numpy array of size [kernel_height, kernel_width].
    # Returns:
    #   a numpy array of size [image_height, image_width] (convolution output).
    
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    print (kernel)
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))   
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
        for y in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()        
    return output

(x_train, y_train), (x_test, y_test) = mnist.load_data()
img = x_train[12] / 255.

img = np.pad(img, 2, mode='constant')
# img = img[0:28][0:28]
# plt.imshow(img, cmap=plt.cm.gray)
# plt.show()

# print (np.shape(img))
# print (np.shape(img[0:28, 0:28]))

# kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
# kernel = np.flipud(np.fliplr(kernel))
kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])

conv_img = np.zeros(shape=(28, 28))
for ii in range(3):
    for jj in range(3):
        # print (ii)
        # print (jj)
        # print (np.shape(img[ii:28+ii, jj:28+jj]))
        
        conv_img += kernel[ii][jj] * img[ii:28+ii, jj:28+jj]
        # print (img[ii:28+ii, jj:28+jj][11, 14])
       
print ("kernel") 
print (kernel)
print ("pixels")
print (img[11:14, 14:17])
print ("result")
print (conv_img[11, 14])

plt.imshow(conv_img, cmap=plt.cm.gray)
plt.show()

