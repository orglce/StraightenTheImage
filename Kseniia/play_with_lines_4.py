import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

toProcess = "../sample_photos/simple_shape_triangle.png"

image = cv2.imread(toProcess)
imageHough = image.copy()
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(grey, 10, 60)
thetaLines = cv2.HoughLines(canny, rho = 1, theta = np.pi / 90, threshold = 55)
lines = cv2.HoughLinesP(canny, rho = 1, theta = np.pi / 90, threshold = 55)

ln = []

if lines is not None:
    for houghLine in lines:
        x1, y1, x2, y2 = houghLine[0]
        
        length = np.round( np.sqrt((x2-x1)**2 + (y2-y1)**2), 1)
        if length > 50:
            theta = np.arctan2(y2-y1, x2-x1) - np.pi/2
            if theta < 0:
                theta = np.pi + theta
                
            theta = np.round(theta * 180 / np.pi, 0)
            ln.append((length,theta))
            imageHough = cv2.line(img = imageHough,
                 pt1 = (x1,y1),
                 pt2 = (x2,y2),
                 color = (255,0,0),
                 thickness = 2)
        


plt.imshow(imageHough)
plt.axis('off')
for line in ln:
    print("length = ",line[0],"; theta = ","{:0.0f}".format(line[1]),' degrees', sep='')
    
for line in thetaLines:
    print("rho = ",line[0][0],"; theta = ","{:0.0f}".format(180*line[0][1]/np.pi),' degrees', sep='')