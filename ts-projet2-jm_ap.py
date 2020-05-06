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
    im = np.copy(img) # On fait une copie de l'original
    for i in range(im.shape[0]):
        for j in range(im.shape[1]):
            r, g, b, dt = im[i, j]
            if(color>4):
                tab=[r*coef[0],g*coef[1],b*coef[2]]
                tab.remove(0)
                r=g=b=min(tab)
            elif(color==4):
                r = g = b = int(0.299 * r + 0.587 * g + 0.114 * b)
            
            im[i,j]=np.multiply((r,g,b,dt),coef)
    return im

def fft(img):
    ft = np.fft.fft(img)
    freq = np.fft.fftfreq(len(img))
    plt.plot(freq, ft.real**2 + ft.imag**2)

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

img = mpimg.imread(getFileName())
if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
    img = (img * 255).astype(np.uint8)

show("RGB","Filtre RGB",img,1)
show("CMY","Filtre CMY",img,5)
show("GREY","Filtre Gris",img,4)
#fft(img)
plt.show()

