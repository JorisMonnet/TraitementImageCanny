from tkinter import filedialog
from tkinter import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import math
import matplotlib.image as mpimg


def getFileName():
    root = Tk()
    fileName = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
    root.destroy()
    return fileName 
 
def non_max_suppression(gradient_magnitude, gradient_direction):
    image_row, image_col = gradient_magnitude.shape
 
    output = np.zeros(gradient_magnitude.shape)
 
    PI = 180
 
    for row in range(1, image_row - 1):
        for col in range(1, image_col - 1):
            direction = gradient_direction[row, col]
 
            if (0 <= direction < PI / 8) or (15 * PI / 8 <= direction <= 2 * PI):
                before_pixel = gradient_magnitude[row, col - 1]
                after_pixel = gradient_magnitude[row, col + 1]
 
            elif (PI / 8 <= direction < 3 * PI / 8) or (9 * PI / 8 <= direction < 11 * PI / 8):
                before_pixel = gradient_magnitude[row + 1, col - 1]
                after_pixel = gradient_magnitude[row - 1, col + 1]
 
            elif (3 * PI / 8 <= direction < 5 * PI / 8) or (11 * PI / 8 <= direction < 13 * PI / 8):
                before_pixel = gradient_magnitude[row - 1, col]
                after_pixel = gradient_magnitude[row + 1, col]
 
            else:
                before_pixel = gradient_magnitude[row - 1, col - 1]
                after_pixel = gradient_magnitude[row + 1, col + 1]
 
            if gradient_magnitude[row, col] >= before_pixel and gradient_magnitude[row, col] >= after_pixel:
                output[row, col] = gradient_magnitude[row, col]
 
    return output
 
 
def threshold(image, low, high, weak,strong):
    newImage = np.zeros(image.shape)
  
    strong_row, strong_col = np.where(image >= high)
    weak_row, weak_col = np.where((image <= high) & (image >= low))
 
    newImage[strong_row, strong_col] = strong
    newImage[weak_row, weak_col] = weak
 
    return newImage
 
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

def convolution(image, kernel):
    if len(image.shape) > 2:
        im = np.zeros([image.shape[0],image.shape[1]])
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                r,g,b,dt = image[i,j]
                grey= int(0.299 * r + 0.587 * g + 0.114 * b)
                im[i,j]=grey
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

def dnorm(x, mu, sd):
    return 1 / (np.sqrt(2 * np.pi) * sd) * np.e ** (-np.power((x - mu) / sd, 2) / 2)

def gaussian_blur(image, kernel_size):
    return convolution(image, gaussian_kernel(kernel_size, sigma=math.sqrt(kernel_size)))

def sobel_edge_detection(image, filter):
    new_image_x = convolution(image, filter)
    new_image_y = convolution(image, np.flip(filter.T, axis=0))

    gradient_magnitude = np.sqrt(np.square(new_image_x) + np.square(new_image_y))

    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    gradient_direction = np.arctan2(new_image_y, new_image_x)

    return gradient_magnitude, gradient_direction
def getImage(imgName):
    img = mpimg.imread(imgName)
    if img.dtype != np.uint8: # Si le résultat n'est pas un tableau d'entiers
        img = (img * 255).astype(np.uint8)
    
    row, col, ch = img.shape

    if ch == 4: #image rgba
        return img

    if ch != 3 :  #image non rgb 
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
if __name__ == '__main__':
 
    image = filterColor(getImage(getFileName()),4)
    weak = 25
    strong = 255

    blurred_image = gaussian_blur(image, kernel_size=2)
  
    gradient_magnitude, gradient_direction = sobel_edge_detection(blurred_image, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
 
    new_image = non_max_suppression(gradient_magnitude, gradient_direction)

    ImageTresHold = threshold(new_image, 5, 25, weak,strong)
 
    CannyImage = hysteresis(ImageTresHold, weak,strong)
 
    plt.imshow(CannyImage, cmap='gray')
    plt.title("Canny Edge Detector")
    plt.show()