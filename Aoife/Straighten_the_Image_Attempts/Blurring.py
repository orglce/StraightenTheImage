import cv2, numpy as np
from matplotlib import pyplot as plt # Good for graphing; install using `pip install matplotlib`
from matplotlib import image as image
#import easygui # An easy-to-use file-picker; pip install easygui
import os
from skimage.transform import probabilistic_hough_line

###### LOOK INTO **************
#It's not a difficult problem you've got yourself there. Without changing much,
# one thing you could try is to play with the "threshold" parameter in the
# HoughLines() OpenCV function, so that only the most prominent lines are returned

#https://stackoverflow.com/questions/30746327/get-a-single-line-representation-for-multiple-close-by-lines-clustered-together

#REf
# https://stackoverflow.com/questions/52365190/blur-a-specific-part-of-an-image
# https://stackoverflow.com/questions/58098161/remove-small-vertical-lines-in-between-the-character-from-an-image
# https://stackoverflow.com/questions/46274961/removing-horizontal-lines-in-image-opencv-python-matplotlib
# https://www.codegrepper.com/code-examples/python/python+plot+vertical+lines
# https://stackoverflow.com/questions/34458251/plot-over-an-image-background-in-python
# https://scikit-image.org/docs/dev/install.html
# https://scikit-image.org/docs/dev/auto_examples/edges/plot_line_hough_transform.html

img = cv2.imread("linestrack.jpg")
blurred_img = cv2.GaussianBlur(img, (21, 21), 0)

mask = np.zeros((183, 275, 3), dtype=np.uint8)
mask = cv2.circle(mask, (50, 50), 100, [255, 255, 255], -1)
cv2.circle

out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)

plt.imshow(out)
plt.show()



# 2
image = cv2.imread('sheetmusiclines.jpg')
mask = np.ones(image.shape, dtype=np.uint8) * 255
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    area = cv2.contourArea(c)
    if area < 1000:
        x,y,w,h = cv2.boundingRect(c)
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]

#plt.imshow(thresh)
#plt.show()
#plt.imshow(mask)
#plt.show()

# sheet music lines
music = cv2.imread('sheetmusiclines.jpg')
musicgray = cv2.cvtColor(music,cv2.COLOR_BGR2GRAY)
musicthresh = cv2.threshold(musicgray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
plt.imshow(musicthresh, cmap=plt.cm.gray)
plt.show()

# Remove horizontal
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25,1))
detected_lines = cv2.morphologyEx(musicthresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    cv2.drawContours(music, [c], -1, (255,255,255), 2)

# Repair image
repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,6))
result = 255 - cv2.morphologyEx(255 - music, cv2.MORPH_CLOSE, repair_kernel, iterations=1)

plt.imshow(musicthresh)
plt.show()
plt.imshow(detected_lines)
plt.show()
plt.imshow(music)
plt.show()
plt.imshow(result)
plt.show()

# 4
#inverse the image, so that lines are black for masking
#img = cv2.bitwise_not(img)
#horizontal = cv2.adaptiveThreshold(img,255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
#horizontal_inv = cv2.bitwise_not(horizontal)
#perform bitwise_and to mask the lines with provided mask
#masked_img = cv2.bitwise_and(img, img, mask=horizontal_inv)
#reverse the image back to normal
#masked_img_inv = cv2.bitwise_not(masked_img)
#plt.imshow(masked_img_inv)
#plt.show()


# 5
xposition = [0.3, 0.4, 0.45]
for xc in xposition:
    plt.axvline(x=xc, color='k', linestyle='--')

# 6
fig, ax = plt.subplots()
x = range(300)
ax.imshow(img, extent=[0, 400, 0, 300])
ax.plot(x, x, '--', linewidth=5, color='firebrick')

# 7

# Line finding using the Probabilistic Hough Transform
image2 =  cv2.imread("linestrack.jpg")
edges = cv2.Canny(image, 2, 1, 25)
lines = probabilistic_hough_line(edges, threshold=10, line_length=5,
                                 line_gap=3)

# Generating figure 2
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
ax = axes.ravel()

ax[0].imshow(image, cmap=plt.cm.gray)
ax[0].set_title('Input image')

ax[1].imshow(edges, cmap=plt.cm.gray)
ax[1].set_title('Canny edges')

ax[2].imshow(edges * 0)
for line in lines:
    p0, p1 = line
    ax[2].plot((p0[0], p1[0]), (p0[1], p1[1]))
ax[2].set_xlim((0, image.shape[1]))
ax[2].set_ylim((image.shape[0], 0))
ax[2].set_title('Probabilistic Hough')

for a in ax:
    a.set_axis_off()

plt.tight_layout()
plt.show()