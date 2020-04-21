"""from tkinter.filedialog import askopenfilename
from tkinter.ttk import *
from tkinter import*

root = Tk() 
root.geometry('200x100') 

def openFile():
    file.name = askopenfilename(initialdir = "/",title = "Select file",filetypes = ("all files",'*.*'))
    if file is not None:
        content = file.read()
    print("réussi")
        #return content

btn = Button(root, text ='Open', command = lambda:openFile()) 
btn.pack(side = TOP, pady = 10)

mainloop()"""

from tkinter import filedialog
from tkinter import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

root = Tk()
root.filename =  filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("png files","*.png"),("all files","*.*")))
img = mpimg.imread(root.filename)
if img.dtype == np.float32: # Si le résultat n'est pas un tableau d'entiers
    img = (img * 255).astype(np.uint8)
plt.imshow(img)
plt.show()
