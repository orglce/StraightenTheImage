import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

expectedNumOfLines = 400
allowedVariance = 100
lowerNumOfLines = expectedNumOfLines - allowedVariance
upperNumOfLines = expectedNumOfLines + allowedVariance

testHowManyPhotos = 24
onlyOneImage = False
imageNameOverride = "../sample_photos/0012.jpg"

for i in range(testHowManyPhotos):

    numOfLines = 0

    bilateralBlurStrength = 7
    bilateralKernelSize = 100

    improveContrast = True
    useBilateralBlur = True

    houghLineTransformThreshold = 50
    houghLineTransformMinLength = 100

    numberOfLoopsAllowed = 30
    loopsCounter = 0

    num = str(format(i + 1, '02d'))
    imageName = '../sample_photos/00' + num + '.jpg'
    if onlyOneImage:
        imageName = imageNameOverride

    # read image
    imgBlack = cv.imread(imageName, cv.IMREAD_GRAYSCALE)
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

        cannyLowTreshod = np.median(blurredImg) - blurredImg.std()
        cannyHighTreshold = np.median(blurredImg) + blurredImg.std()
        cannyAperture = 3
        edges = cv.Canny(blurredImg, cannyLowTreshod, cannyHighTreshold, cannyAperture)

        lines = cv.HoughLinesP(edges, 1, np.pi / 180, houghLineTransformThreshold,
                               minLineLength=houghLineTransformMinLength, maxLineGap=50)
        numOfLines = len(lines)

        print(numOfLines)

        if loopsCounter > numberOfLoopsAllowed:
            break

        loopsCounter = loopsCounter + 1

        for line in lines:
            for x1, y1, x2, y2 in line:
                cv.line(imgBlackCopy, (x1, y1), (x2, y2), (255, 0, 0), 2)

        cv.imwrite("lines" + str(loopsCounter + 1) + ".png", imgBlackCopy)
        cv.imwrite("edges" + str(loopsCounter + 1) + ".png", edges)



    print(imageName)
    print("Number of lines:", numOfLines)
    print("Bilateral: ", bilateralBlurStrength)


# plot images
fig, axs = plt.subplots(2, 2)
axs[0, 0].imshow(imgBlack, "gray")
axs[0, 0].set_title('Original B&W')
axs[0, 1].imshow(blurredImg, 'gray')
axs[0, 1].set_title('Bilateral blur')
axs[1, 0].imshow(edges, 'gray')
axs[1, 0].set_title('Canny Edge')
axs[1, 1].imshow(cv.cvtColor(imgBlackCopy, cv.COLOR_BGR2RGB))
axs[1, 1].set_title('Hough lines')

plt.show()
