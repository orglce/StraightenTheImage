#Simple Program to detect the edges on a simple image and straighten it
#Image Processing Project

import cv2, numpy as np
from matplotlib import pyplot as plt # Good for graphing; install using `pip install matplotlib`
from matplotlib import image as image
#import easygui # An easy-to-use file-picker; pip install easygui
import os

#Open Cv uses BGR and matplotlib uses RGB

I = cv2.imread("IPTest2.png",1)
image = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
plt.imshow(image)
plt.show()