# Will Blur and detect a try to detect a single line of horizon picture else it will more onto the general algorithm
# Things to Note
# From Testing the image seems to come out better/ detect the horizon better when converted to grayscale first

import cv2, numpy as np
from matplotlib import pyplot as plt # Good for graphing; install using `pip install matplotlib`
from matplotlib import image as image
#import easygui # An easy-to-use file-picker; pip install easygui
import os
from skimage.transform import probabilistic_hough_line
blur_value = 5

# Original Image
image = cv2.imread("H5.jpg", 1)
#Convert to RGB
img2 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(img2)
plt.show()

first_image_copy = img2.copy()
second_image_copy = img2.copy()
another_copy = image.copy()
copy_gray = cv2.cvtColor(another_copy, cv2.COLOR_BGR2GRAY)

# Canny before image is blurred to see if there is a difference
edges = cv2.Canny(img2,100, 200)
#plt.imshow(edges, cmap = plt.cm.gray)
#plt.show()
#cv2.imwrite("Canny_Before_Blur.jpg", edges)

# Hough
def printLinesToImage(img_P, lines):
    for x in range(0, len(lines)):
        for rho, theta in lines[x]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            cv2.line(img_P, (x1, y1), (x2, y2), (0, 255, 0), 2)

#blurred image
def blur_image(img_B, blur):
    blurred_img_gray = cv2.GaussianBlur(img_B, (blur, blur), 0)
    plt.imshow(blurred_img_gray, cmap = plt.cm.gray)
    plt.show()
    #cv2.imwrite("Grayscale_blur.jpg", blurred_img_gray)
    return blurred_img_gray

#Canny on grayscale image
def Canny(img_C):
    edges_gray = cv2.Canny(img_C, 99, 100)
    plt.imshow(edges_gray, cmap = plt.cm.gray)
    plt.show()
    #cv2.imwrite("Canny_on_gray.jpg", edges_gray)
    return edges_gray

blur_image_res = blur_image(copy_gray, blur_value)
gray_edges_result = Canny(blur_image_res)

#print(gray_edges_result)

#for e in edges_gray:
#    print(e)


def finish(img_F, l):
    #for line in l:
    #    print(line)
    #    print(len(l))
    printLinesToImage(img_F, l)
    plt.imshow(img_F)
    plt.show()
    second = cv2.cvtColor(img_F, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("Result.jpg",second)


lines = cv2.HoughLines(gray_edges_result,1 , np.pi / 180, 100, 255)
#print(len(lines))
if(lines == None):
    while lines == None:
        blur_value -= 2
        blur_image_res = blur_image(copy_gray, blur_value)
        gray_edges_result = Canny(blur_image_res)
        lines = cv2.HoughLines(gray_edges_result, 1, np.pi / 180, 100, 255)
        if(len(lines) != 0):
            break;
        print("forever in 1")
#elif(len(lines) > 1):
#    while(len(lines) > 1 and len(lines) < 200):
#        blur_value = blur_value + 2
#        blur_image_res = blur_image(copy_gray, blur_value)
#        gray_edges_result = Canny(blur_image_res)
#        lines = cv2.HoughLines(gray_edges_result, 1, np.pi / 180, 0, 255)
#        print("forever in 2")

finish(img2, lines)


