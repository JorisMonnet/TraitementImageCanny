from tkinter import filedialog
from tkinter import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def getFileName():
    root = Tk()
    fileName = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
    root.destroy()
    return fileName

def filterColor(img,color):
    switcher = {
        1: (1, 0, 0, 1),    #r
        2: (0, 1, 0, 1),    #g
        3: (0, 0, 1, 1),    #b
        4: (1, 1, 1, 1),    #grey
        5: (0, 0, 0, 1),    #c
        6: (0, 0, 0, 1),    #m
        7: (0, 0, 0, 1)     #y
        
    }
    coef=switcher.get(color)
    im = np.copy(img) # On fait une copie de l'original
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            r, g, b, dt = im[i, j]
            if(color==4):
                r=g=b = int(0.299 * r + 0.587 * g + 0.114 * b)
            im[i,j]=np.multiply((r,g,b,dt),coef)
    return im
def fft(img):
    return np.fft.fft(img)

"""
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    return g

def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    
    for i in range(1,M-1):
        for j in range(1,N-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle pi/4
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle pi/2
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 3pi/4
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    Z[i,j] = img[i,j]
                else:
                    Z[i,j] = 0

            except IndexError as e:
                pass
    
    return Z
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    M, N = img.shape
    res = np.zeros((M,N), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    
    return (res, weak, strong)

def hysteresis(img, weak, strong=255):
    M, N = img.shape  
    for i in range(1, M-1):
        for j in range(1, N-1):
            if (img[i,j] == weak):
                try:
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img
"""
"""img = mpimg.imread(getFileName())
if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
    img = (img * 255).astype(np.uint8)"""

imgpil = Image.open(getFileName())  
img = np.array(imgpil) # Transformation de l'image en tableau numpy

fig = plt.figure("RGB")
fig.suptitle("Filtre RGB")

ax = fig.add_subplot(2,2,1)
ax.imshow(img)
ax.set_xticklabels([])
ax.set_yticklabels([])
for i in range(1,4):
    ax = fig.add_subplot(2,2,i+1)
    ax.imshow(filterColor(img,i))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
fig.show()

fig2 = plt.figure("GREY")
fig2.suptitle("GREY")

ax2 = fig2.add_subplot(1,2,1)
ax2.imshow(img)
ax2.set_xticklabels([])
ax2.set_yticklabels([])
ax2 = fig2.add_subplot(1,2,2)
ax2.imshow(filterColor(img,4))
ax2.set_xticklabels([])
ax2.set_yticklabels([])
fig2.show()

fig3 = plt.figure("CMY")
fig3.suptitle("Filtre CMY")

ax3 = fig3.add_subplot(2,2,1)
ax3.imshow(img)
ax3.set_xticklabels([])
ax3.set_yticklabels([])
for i in range(5,8):
    ax3 = fig3.add_subplot(2,2,i-3)
    ax3.imshow(filterColor(img,i))
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
fig3.show()
"""
fig4 = plt.figure("FFT")
fig4.suptitle("FFT")

ax4 = fig4.add_subplot(3,1,1)
ax4.imshow(img)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
B = fft(img)
ax4 = fig4.add_subplot(3,1,2)
ax4.plot(np.real(B))
ax4.ylabel("partie reelle")
ax4 = fig4.add_subplot(3,1,2)
ax4.plot(np.imag(B))
ax4.ylabel("partie imaginaire")
fig4.show()"""