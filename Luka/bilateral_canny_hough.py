import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

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
	print(len(lines))

imageName = '../sample_photos/0001.jpg'


bilateralBlur = True
bilateralBlurStrength = 4


# read image
imgBlack = cv.imread(imageName, cv.IMREAD_GRAYSCALE)

width = imgBlack.shape[1]
height = imgBlack.shape[0]
#
if width > height:
	newHeight = int(1000*height/width)
	imgBlack = cv.resize(imgBlack, (1000, newHeight), cv.INTER_AREA)

else:
	newWidth = int(1000*width/height)
	imgBlack = cv.resize(imgBlack, (newWidth, 1000), cv.INTER_AREA)

imgBlack = cv.equalizeHist(imgBlack)
clahe = cv.createCLAHE(2, (8, 8))
imgBlack = clahe.apply(imgBlack)
# apply bilateral blur filter
# idk what other two parameters do yet
if (bilateralBlur):
	blurredImg = cv.bilateralFilter(imgBlack, bilateralBlurStrength, 100, 100)
else:
	blurredImg = imgBlack

cannyLowTreshod = 50
cannyHighTreshold = 100
cannyAperture = 3

cannyLowTreshod = blurredImg.mean()-blurredImg.std()
cannyHighTreshold = blurredImg.mean()+blurredImg.std()
cannyAperture = 3

print(cannyLowTreshod, cannyHighTreshold)

# apply canny edge detection on the blurred image
edges = cv.Canny(blurredImg, cannyLowTreshod, cannyHighTreshold, cannyAperture)

# find lines from the edges
lines = cv.HoughLines(edges, 1, np.pi/180, 100)
printLinesToImage(imgBlack, lines)

# plot images
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(blurredImg, "gray")
# axs[0, 0].set_title('Original')
axs[0, 1].imshow(blurredImg, 'gray')
# axs[0, 1].set_title('Bilateral blur')
axs[1, 0].imshow(edges, 'gray')
# axs[1, 0].set_title('Canny Edge')
axs[1, 1].imshow(cv.cvtColor(imgBlack, cv.COLOR_BGR2RGB))
# axs[1, 1].set_title('Hough lines')

plt.show()
