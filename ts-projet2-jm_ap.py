from tkinter import filedialog
from tkinter import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

"""
gaussian kernel, useful to blur an image
"""
def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    return  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal

"""
allow to convolve the gaussian kernel and an image to blur it
"""
def convolution(image, kernel):
    if len(image.shape) > 2:
        im = np.zeros([image.shape[0],image.shape[1]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r,g,b,dt = image[i,j]
                im[i,j]= int(0.299 * r + 0.587 * g + 0.114 * b)
        image = np.asarray(im,dtype = "uint8")
    image_row, image_col = image.shape
    kernel_row, kernel_col = kernel.shape

    output = np.zeros(image.shape)

    pad_height = int((kernel_row - 1) / 2)
    pad_width = int((kernel_col - 1) / 2)

    padded_image = np.zeros((image_row + (2 * pad_height), image_col + (2 * pad_width)))

    padded_image[pad_height:padded_image.shape[0] - pad_height, pad_width:padded_image.shape[1] - pad_width] = image

    for row in range(image_row):
        for col in range(image_col):
            output[row, col] = np.sum(kernel * padded_image[row:row + kernel_row, col:col + kernel_col])

    return output

"""
return an image blured with a gaussain kernel
"""
def gaussian_blur(image, kernel_size):
    return convolution(image, gaussian_kernel(kernel_size))

"""
detect the edge intensity and direction by calculating the gradient
"""
def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)
    
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)
    
    gradient = np.hypot(Ix, Iy)
    gradient = gradient / gradient.max() * 255
    theta = np.arctan2(Iy, Ix)
    
    return (gradient, theta)

"""
allow to thin out the edges
"""
def non_max_suppression(img, gradient_direction):
    row, col = img.shape
 
    result = np.zeros((img.shape),dtype=np.int32)
 
    angle = gradient_direction * 180. / np.pi
    angle[angle < 0] += 180
 
    for i in range(1,row-1):
        for j in range(1,col-1):
            try:
                q = 255
                r = 255
                
               #angle 0
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = img[i, j+1]
                    r = img[i, j-1]
                #angle 45
                elif (22.5 <= angle[i,j] < 67.5):
                    q = img[i+1, j-1]
                    r = img[i-1, j+1]
                #angle 90
                elif (67.5 <= angle[i,j] < 112.5):
                    q = img[i+1, j]
                    r = img[i-1, j]
                #angle 135
                elif (112.5 <= angle[i,j] < 157.5):
                    q = img[i-1, j-1]
                    r = img[i+1, j+1]

                if (img[i,j] >= q) and (img[i,j] >= r):
                    result[i,j] = img[i,j]
                else:
                    result[i,j] = 0

            except IndexError as e:
                pass
    
    return result

"""
identify the pixel as strong (they contribute to edges) 
or weak (they maybe contribute) 
or non relevant (they don't contribute)
"""
def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio
    
    result = np.zeros((img.shape), dtype=np.int32)
    
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)
    
    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
    
    result[strong_i, strong_j] = strong
    result[weak_i, weak_j] = weak
    
    return (result, weak, strong)

"""
allow weak pixels to become strong if there is others strong pixel near them
"""
def hysteresis(img, weak, strong):
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
use all the previous functions to create an image of the contoutrs of a source image via canny
"""
def canny(img):
    image = filterColor(img,4)

    blurred_image = gaussian_blur(image, kernel_size=4)
  
    imageSobel, gradient_direction = sobel_filters(blurred_image)
 
    imageNonMax = non_max_suppression(imageSobel, gradient_direction)
    
    imageTresHold, weak, strong = threshold(imageNonMax)
    
    return hysteresis(imageTresHold, weak, strong)

"""
show the canny image
param = image source
"""
def showCanny(img):
    plt.figure("Canny Edge Detector")
    showSubPlot(211,img)
    showSubPlot(212,canny(img),True)

"""
show a canny image with the cv2 library to compare
work only with the testImage here
"""
def cannyCV2():
    import cv2 as cv
    im = cv.imread('imgTest.png',0)
    ret, thresh = cv.threshold(im, 245, 255, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    plt.figure("Canny via cv2")
    plt.imshow(cv.drawContours(im, contours, -1, (0,0,0), 1),cmap="gray")

"""
filter the colors of an image and return the result
param = image source 
color index : 1 = red; 2 = green; 3 = blue; 4 = grey
5 = cyan;  6 = magenta; 7 = yellow
"""
def filterColor(img,color):
    switcher = {
        1: (1, 0, 0, 1),    #r
        2: (0, 1, 0, 1),    #g
        3: (0, 0, 1, 1),    #b
        4: (1, 1, 1, 1),    #grey
        5: (1, 1, 0, 1),    #c
        6: (1, 0, 1, 1),    #m
        7: (0, 1, 1, 1)     #y
    }

    coef=switcher.get(color)
    im = np.copy(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            try :
                r, g, b, dt = im[i, j]
            except:
                r,g,b = im[i,j]
                dt=1
            if(color>4):
                tab=[r*coef[0],g*coef[1],b*coef[2]]
                tab.remove(0)
                r=g=b=min(tab)
            elif(color==4):
                r = g = b = int(0.299 * r + 0.587 * g + 0.114 * b)
            im[i,j]=np.multiply((r,g,b,dt),coef)
    return im

"""
matplotlib issue in imshow: https://github.com/matplotlib/matplotlib/issues/9391/
this function show the original image , the spectrum centered of the fast fourier transform,
and then recreate the image from the spectrum
"""
def showImagefft(img,name="Fast Fourier Transform"):
    plt.figure(name)
    fshift = np.fft.fftshift(np.fft.fft2((img * 255).astype(np.uint8))) #image is shifted to center
    newImage = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    plt.subplot(131),plt.imshow(img,cmap="gray")
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow((np.abs(fshift) * 255).astype(np.uint8),cmap="gray")
    plt.title('Spectrum via FFT'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow((newImage * 255).astype(np.uint8),cmap="gray")
    plt.title('Reconstitued Image'), plt.xticks([]), plt.yticks([])

"""
show a figure with a colorFiltered image
"""
def show(title,suptitle,img,index):
    plt.figure(title)
    plt.suptitle(suptitle)
    showSubPlot(221,img)
    showSubPlot(222,filterColor(img,index))
    if(index!=4):
        showSubPlot(223,filterColor(img,index+1))
        showSubPlot(224,filterColor(img,index+2))

"""
remove axes on the subplot and show the image
"""
def showSubPlot(index,img,cmapGray=False):
    plt.subplot(index)
    if cmapGray:
        a=plt.imshow(img,cmap="gray")
    else:
        a=plt.imshow(img)
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)

"""
open an image from a filedialog at the rgba format
"""
def getImage():
    root = Tk()
    fileName = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
    root.destroy()
    img = mpimg.imread(fileName)
    if img.dtype != np.uint8: # Si le résultat n'est pas un tableau d'entiers
        img = (img * 255).astype(np.uint8)
    
    row, col, ch = img.shape

    if ch == 4: #image rgba
        return img

    if ch != 3 :  #image non rgb ni rgba"
        raise Exception("Bad Image Type")
    """
    If the image is only rgb, by adding the opacity, the blank around the images is used into the filtering of colors and 
    the other functionnalities of this project, try to avoid rgb images and prefer rgba to have a better view of the functionnalities
    """
    rgb = np.zeros([row,col,4])
    for i in range(row):
        for j in range(col):
            r,g,b = img[i,j]
            rgb[i,j]=r,g,b,255

    return np.asarray(rgb, dtype='uint8')

"""
it takes a long time to process all the filters for HD images
"""
if __name__ == "__main__":
    img = getImage()
    show("RGB","Filtre RGB",img,1)
    show("CMY","Filtre CMY",img,5)
    show("GREY","Filtre Gris",img,4)
    showImagefft(filterColor(img,4)) 
    showImagefft(canny(img),"FFT after canny") #compare after canny, change the name of the figure for each fft launched together
    showCanny(img)
    cannyCV2()
    plt.show()

"""
Sources : 
Canny : https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
fft : https://numpy.org/doc/stable/reference/generated/numpy.fft.fft2.html
"""