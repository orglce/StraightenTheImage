import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image

toProcess = "../sample_photos/simple_shape_rectangle.png"
image = cv2.imread(toProcess)
edges = cv2.Canny(image, 100,150)
plt.imshow(edges, cmap='gray')
cv2.imwrite('simple_edge.png', edges)

