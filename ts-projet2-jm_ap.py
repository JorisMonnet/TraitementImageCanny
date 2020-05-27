from tkinter import filedialog
from tkinter import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
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
matplotlib issue in imshow:
https://github.com/matplotlib/matplotlib/issues/9391/
"""
def showImagefft(img):
    plt.figure()
    fshift = np.fft.fftshift(np.fft.fft2((img * 255).astype(np.uint8))) #image is shifted to center
    newImage = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift)))
    plt.subplot(131),plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(132),plt.imshow((np.abs(fshift) * 255).astype(np.uint8),cmap="gray")
    plt.title('Spectrum via FFT'), plt.xticks([]), plt.yticks([])
    plt.subplot(133),plt.imshow((newImage * 255).astype(np.uint8))
    plt.title('Reconstitued Image'), plt.xticks([]), plt.yticks([])

def show(title,suptitle,img,index):
    plt.figure(title)
    plt.suptitle(suptitle)
    showSubPlot(221,img)
    showSubPlot(222,filterColor(img,index))
    if(index!=4):
        showSubPlot(223,filterColor(img,index+1))
        showSubPlot(224,filterColor(img,index+2))

def showSubPlot(index,img):
    plt.subplot(index)
    a=plt.imshow(img)
    a.axes.get_xaxis().set_visible(False)
    a.axes.get_yaxis().set_visible(False)

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
            if(r>5 or g>5 or b>5):
                rgb[i,j]=r,g,b,255

    return np.asarray(rgb, dtype='uint8')

if __name__ == "__main__":
    imgName = getFileName()
    #show("RGB","Filtre RGB",getImage(imgName),1)
    #show("CMY","Filtre CMY",getImage(imgName),5)
    #show("GREY","Filtre Gris",getImage(imgName),4)
    showImagefft(filterColor(getImage(imgName),4))     

    plt.show()

