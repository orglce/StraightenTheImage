import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def getLines(img, expectedNumOfLines, numOfLinesVariance):

    # range for number of lines that we wish to detect
    lowerNumOfLines = expectedNumOfLines - numOfLinesVariance
    upperNumOfLines = expectedNumOfLines + numOfLinesVariance

    # starting parameters for the line detection
    numOfLines = 0

    bilateralBlurStrength = 7

    improveContrast = True
    useBilateralBlur = True

    houghLineTransformThreshold = 50
    houghLineTransformMinLength = 100

    # in case of infinite looping
    numberOfLoopsAllowed = 30
    loopsCounter = 0

    # reading and resizing the image to speed up further processing
    imgBlack = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    width = imgBlack.shape[1]
    height = imgBlack.shape[0]

    if width > height:
        newHeight = int(1000 * height / width)
        imgBlack = cv.resize(imgBlack, (1000, newHeight), cv.INTER_AREA)
    else:
        newWidth = int(1000 * width / height)
        imgBlack = cv.resize(imgBlack, (newWidth, 1000), cv.INTER_AREA)

    while numOfLines < lowerNumOfLines or numOfLines > upperNumOfLines:
        imgBlackCopy = imgBlack.copy()

        if numOfLines < lowerNumOfLines:
            bilateralBlurStrength = bilateralBlurStrength - 1
        elif numOfLines > upperNumOfLines:
            bilateralBlurStrength = bilateralBlurStrength + 1

        if bilateralBlurStrength > 20:
            houghLineTransformMinLength += 50
        elif bilateralBlurStrength > 15:
            improveContrast = False
        elif bilateralBlurStrength == 0:
            useBilateralBlur = False

        if improveContrast:
            imgBlackCopy = cv.equalizeHist(imgBlackCopy)
            imgBlackCopy = cv.createCLAHE(1, (5, 5)).apply(imgBlackCopy)

        if useBilateralBlur:
            blurredImg = cv.bilateralFilter(imgBlackCopy, bilateralBlurStrength, 100, 100)
        else:
            blurredImg = imgBlackCopy

        # Canny edge detection
        cannyLowThreshold = np.median(blurredImg) - blurredImg.std()
        cannyHighThreshold = np.median(blurredImg) + blurredImg.std()
        imgEdges = cv.Canny(blurredImg, cannyLowThreshold, cannyHighThreshold, 3)

        # Hough line transform
        houghLines = cv.HoughLinesP(imgEdges, 1, np.pi / 180, houghLineTransformThreshold,
                               minLineLength=houghLineTransformMinLength, maxLineGap=50)
        numOfLines = len(houghLines)

        if loopsCounter > numberOfLoopsAllowed:
            break
        loopsCounter = loopsCounter + 1

    for line in houghLines:
        for x1, y1, x2, y2 in line:
            cv.line(imgBlackCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)

    return imgBlackCopy, imgEdges, houghLines


img = cv.imread("./../sample_photos/0005.jpg")
linesImg, edgesImg, houghLinesP = getLines(img, 300, 100)

fig, axs = plt.subplots(1, 2)
axs[0].imshow(linesImg, "gray")
axs[0].set_title('Lines')
axs[1].imshow(edgesImg, 'gray')
axs[1].set_title('Edges')
plt.show()
