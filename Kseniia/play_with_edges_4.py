import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

toProcess = "./bilateral_filter.jpg"
image = cv2.imread(toProcess)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#plt.imshow(image, cmap = 'gray')
values = image.ravel()

plt.hist(values, bins=range(0,256))