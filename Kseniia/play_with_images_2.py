import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

toProcess = "../sample_photos/car.jpg"
image = cv2.imread(toProcess)
edges = cv2.Canny(image, 100,150)

toProcess = "../sample_photos/car_small.jpg"
image = cv2.imread(toProcess)
grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

edges2 = cv2.Canny(grey, 100,150)
edges3 = cv2.Canny(image, 100,150)

fig, axs = plt.subplots(3, 1, gridspec_kw = {'wspace':0, 'hspace':0.01})

axs[0].imshow(edges, cmap='gray')
axs[0].axis('off')
axs[1].imshow(edges2, cmap='gray')
axs[1].axis('off')
axs[2].imshow(edges3, cmap='gray')
axs[2].axis('off')

cv2.imwrite('huge_edges.jpg', edges)
cv2.imwrite('grey_edges.jpg', edges2)
cv2.imwrite('coulor_edge.jpg', edges3)