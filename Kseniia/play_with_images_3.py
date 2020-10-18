import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

toProcess = "../sample_photos/tipsy_house.jpg"
image = cv2.imread(toProcess)
bilateralBlurStrength = 9
blurredImg = cv2.bilateralFilter(image, bilateralBlurStrength, 150, 150)

edges = cv2.Canny(image, 100,150)
edges2 = cv2.Canny(blurredImg, 100,150)

fig, axs = plt.subplots(2, 1, gridspec_kw = {'wspace':0, 'hspace':0.01})

axs[0].imshow(edges, cmap='gray')
axs[0].axis('off')
axs[1].imshow(edges2, cmap='gray')
axs[1].axis('off')

cv2.imwrite('no_filter.jpg', edges)
cv2.imwrite('bilateral_filter.jpg', edges2)
