# Going to try minimise other lines by blurring before canny and hough
import cv2, numpy as np
from matplotlib import pyplot as plt # Good for graphing; install using `pip install matplotlib`
from matplotlib import image as image
#import easygui # An easy-to-use file-picker; pip install easygui
import os
from skimage.transform import probabilistic_hough_line


img = cv2.imread("horizon.jpg")
img2 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()

# Canny before image is blurred to see if there is a difference
edges = cv2.Canny(img2,100, 200)
plt.imshow(edges)
plt.show()


#blurred image
blurred_img = cv2.GaussianBlur(img2, (5, 5), 0)
plt.imshow(blurred_img)
plt.show()

edges2 = cv2.Canny(blurred_img, 50, 200)
plt.imshow(edges2)
plt.show()

#Note seems to be successfull although the inputs into the blur
# have to be a specific value in order to detect some lines at (20, 20)
# under nothing was dectected