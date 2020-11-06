import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

cannyImage = "./bilateral_filter.jpg"
toProcess = "../sample_photos/tipsy_house.jpg"
canny = cv2.imread(cannyImage)
image = cv2.imread(toProcess)

grey = cv2.cvtColor(canny, cv2.COLOR_BGR2GRAY)
imageHough = image.copy()
lines = cv2.HoughLines(grey, rho = 1, theta = np.pi / 180, threshold = 430)
if lines is not None:
    for houghLine in lines:
        rho = houghLine[0][0]
        theta = houghLine[0][1]
        x1 = 0
        x2 = image.shape[1]
        # vertical line, x1 = x2 = rho, and y1-y2 is just vertical line from top to bottom
        if np.round(np.sin(theta),5) == 0:
            x1 = rho
            y1 = 0
            x2 = rho
            y2 = image.shape[0]
        else:
            y1 = int( ( rho - x1 * np.round(np.cos(theta),5) ) / np.round(np.sin(theta),5) ) 
            y2 = int( ( rho - x2 * np.round(np.cos(theta),5) ) / np.round(np.sin(theta),5) ) 

        imageHough = cv2.line(img = imageHough,
                 pt1 = (x1,y1),
                 pt2 = (x2,y2),
                 color = (0,0,255),
                 thickness = 2)
        
plt.imshow(cv2.cvtColor(imageHough,cv2.COLOR_BGR2RGB))   
plt.axis('off')