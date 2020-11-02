import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

paths = "../sample_photos"
toProcess = ["simple_shape_rectangle.png",
             "simple_shape_cross.png",
             "simple_shape_sector.png",
             "simple_shape_crescent.png",
             "simple_shape_heart.png",
             "simple_shape_triangle.png",
             "simple_shape_wave.png",
             "simple_shape_capsule.png"]


for pic in toProcess:
    image = cv2.imread(paths+"/"+pic)
    imageHough = image.copy()
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(grey, 10, 60)
    lines = cv2.HoughLines(canny, rho = 1, theta = np.pi / 90, threshold = 55)
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
                     color = (255,0,0),
                     thickness = 2)
           
    fig, axs = plt.subplots(1,3, figsize=(20,10), gridspec_kw={'wspace': 0, 'hspace': 0})
    axs[0].imshow(image)
    axs[1].imshow(canny, cmap='gray')
    axs[2].imshow(imageHough)
    axs[0].set_title('Original')
    axs[1].set_title('Canny edges')
    axs[2].set_title('Hough lines')
    axs[0].axis('off')
    axs[1].axis('off')
    axs[2].axis('off')
