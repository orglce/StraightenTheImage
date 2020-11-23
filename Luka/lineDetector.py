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
        imgBlackCopyHoughLinesP = imgBlack.copy()

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
            imgBlackCopyHoughLinesP = cv.equalizeHist(imgBlackCopyHoughLinesP)
            imgBlackCopyHoughLinesP = cv.createCLAHE(1, (5, 5)).apply(imgBlackCopyHoughLinesP)

        if useBilateralBlur:
            blurredImg = cv.bilateralFilter(imgBlackCopyHoughLinesP, bilateralBlurStrength, 100, 100)
        else:
            blurredImg = imgBlackCopyHoughLinesP

        # Canny edge detection
        cannyLowThreshold = np.median(blurredImg) - blurredImg.std()
        cannyHighThreshold = np.median(blurredImg) + blurredImg.std()
        imgEdges = cv.Canny(blurredImg, cannyLowThreshold, cannyHighThreshold, 3)

        # Hough line p transform
        houghLinesP = cv.HoughLinesP(imgEdges, 1, np.pi / 180, houghLineTransformThreshold,
                               minLineLength=houghLineTransformMinLength, maxLineGap=50)
        numOfLines = len(houghLinesP)

        if loopsCounter > numberOfLoopsAllowed:
            break
        loopsCounter = loopsCounter + 1

    # Normal hough line transform
    imgBlackCopyHoughLines = imgBlack.copy()
    houghLines = cv.HoughLines(imgEdges, 1, np.pi / 180, 200)

    # drawing lines
    for x in range(0, len(houghLines)):
        for rho, theta in houghLines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv.line(imgBlackCopyHoughLines, (x1, y1), (x2, y2), (255, 0, 0), 2)

    for line in houghLinesP:
        for x1, y1, x2, y2 in line:
            cv.line(imgBlackCopyHoughLinesP, (x1, y1), (x2, y2), (255, 0, 0), 2)

    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(imgBlackCopyHoughLinesP, "gray")
    axs[0].set_title('HoughLinesP')
    axs[1].imshow(imgBlackCopyHoughLines, 'gray')
    axs[1].set_title('HoughLines normal')
    axs[2].imshow(imgEdges, 'gray')
    axs[2].set_title('Edges')
    plt.show()

    return houghLinesP, houghLines


img = cv.imread("./../sample_photos/0006.jpg")
houghLinesP, houghLines = getLines(img, 400, 100)

# lines for further processing are saved in
# houghLinesP and houghLines


