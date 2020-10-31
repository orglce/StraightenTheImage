#Simple Program to detect the edges on a simple image and straighten it
#Image Processing Project

import cv2, numpy as np
from matplotlib import pyplot as plt # Good for graphing; install using `pip install matplotlib`
from matplotlib import image as image
#import easygui # An easy-to-use file-picker; pip install easygui
import os
import math

#References
# https://www.youtube.com/watch?v=OchCsSiffeE

#Read in Image and show it
I = cv2.imread("IPTest10.jpeg", 1)
image = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
plt.imshow(image)
plt.show()

#Detect the Lines
canimg = cv2.Canny(image, 50, 200)
lines = cv2.HoughLines(canimg, 1, np.pi/180, 120,np.array([]))

for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b= np.sin(theta)

    x0 = a*rho
    y0 = b*rho

    x1 = int(x0+1000*(-b))
    y1 = int(y0-+1000*(a))
    x2 = int(x0-1000*(-b))
    y2 = int(y0-1000*(1))

    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
    slope = (y2-y1)/(x2-x1)

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    angle = np.rad2deg(np.arctan(slope))
    print(angle)
    M = cv2.getRotationMatrix2D((cX, cY), 30, 1.0)
    R = cv2.warpAffine(image, M, (w, h))

plt.imshow(canimg)
plt.show()

# Rotating Manually - what we want it to look like
(h, w) = image.shape[:2]
(cX, cY) = (w // 2, h // 2)
#angle is a simple 360 turn 90 is right angle, 180 upside down image
M = cv2.getRotationMatrix2D((cX, cY),30,1.0)
R = cv2.warpAffine(image, M, (w, h))
plt.xlabel('x axis - Manual Image Rotation')
plt.ylabel('y axis - Manual Image Rotation')
plt.imshow(R)
plt.show()


