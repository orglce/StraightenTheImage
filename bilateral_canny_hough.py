import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

# found online, idk if it works good enough
# we will probably have to implement canny fine tuning ourselves
def autoCanny(image, sigma=0.33):
	v = np.median(image)
	
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv.Canny(image, lower, upper)

	print("Lower: ", lower, "", "Upper: ", upper)

	return edged
	
# prints lines from hough line transform to image
def printLinesToImage(img, lines): 
	for x in range(0, len(lines)):
		for rho,theta in lines[x]:
			a = np.cos(theta)
			b = np.sin(theta)
			x0 = a*rho
			y0 = b*rho
			x1 = int(x0 + 1000*(-b))
			y1 = int(y0 + 1000*(a))
			x2 = int(x0 - 1000*(-b))
			y2 = int(y0 - 1000*(a))

			cv.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

imageName = 'sample_photos/car_small.jpg'

bilateralBlur = True
bilateralBlurStrength = 3

cannyLowTreshod = 50
cannyHighTreshold = 150
cannyAperture = 3

# read image
origImg = cv.imread(imageName)
img = origImg.copy();

# convert to black and white
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# apply bilateral blur filter
# idk what other two parameters do yet
if (bilateralBlur):
	blurredImg = cv.bilateralFilter(img, bilateralBlurStrength, 100, 100)
else:
	blurredImg = img

# apply canny edge detection on the blurred image
edges = cv.Canny(blurredImg, cannyLowTreshod, cannyHighTreshold, cannyAperture)

# find lines from the edges
lines = cv.HoughLines(edges, 1, np.pi/180, 200)
printLinesToImage(img, lines)

# plot images
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(cv.cvtColor(origImg, cv.COLOR_BGR2RGB))
axs[0, 0].set_title('Original')
axs[0, 1].imshow(blurredImg, 'gray')
axs[0, 1].set_title('Bilateral blur')
axs[1, 0].imshow(edges, 'gray')
axs[1, 0].set_title('Canny Edge')
axs[1, 1].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
axs[1, 1].set_title('Hough lines')

plt.show()
