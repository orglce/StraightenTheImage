import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as image


toProcess = "../sample_photos/simple_shape_rectangle.png"

image = cv2.imread(toProcess)




h, w, d = image.shape

cx = w // 2
cy = h // 2
d = 30
s = 1

# Rotate and expand
M = cv2.getRotationMatrix2D(center = (cx,cy), angle = d, scale = s)

# rotate expanding bounds of image to not cut it off
# cos and sin of the rotation's angle can be taken from matrix
cost = np.abs(M[0,0])
sint = np.abs(M[0,1])

# get new width and height using polar to cartesian translation 
nw = int(h * sint + w * cost)
nh = int(h * cost + w * sint)

# Adjust matrix to move rotated image and make sure any part is not outside
M[0,2] = M[0,2] + (nw / 2) - cx
M[1,2] = M[1,2] + (nh / 2) - cy

# Details taken from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/

R1 = cv2.warpAffine(image, M = M, dsize = (nw, nh))


# Rotate and keep

M = cv2.getRotationMatrix2D(center = (cx,cy), angle = d, scale = s)
R2 = cv2.warpAffine(image, M = M, dsize = (w, h))
#R2 = cv2.resize(R2, dsize = (2 * w, 2 * h))

# https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders

def rotatedRectWithMaxArea(w, h, angle):
  """
  Given a rectangle of size wxh that has been rotated by 'angle' (in
  radians), computes the width and height of the largest possible
  axis-aligned rectangle (maximal area) within the rotated rectangle.
  """
  if w <= 0 or h <= 0:
    return 0,0

  width_is_longer = w >= h
  side_long, side_short = (w,h) if width_is_longer else (h,w)

  # since the solutions for angle, -angle and 180-angle are all the same,
  # if suffices to look at the first quadrant and the absolute values of sin,cos:
  sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
  if side_short <= 2.*sin_a*cos_a*side_long or abs(sin_a-cos_a) < 1e-10:
    # half constrained case: two crop corners touch the longer side,
    #   the other two corners are on the mid-line parallel to the longer line
    x = 0.5*side_short
    wr,hr = (x/sin_a,x/cos_a) if width_is_longer else (x/cos_a,x/sin_a)
  else:
    # fully constrained case: crop touches all 4 sides
    cos_2a = cos_a*cos_a - sin_a*sin_a
    wr,hr = (w*cos_a - h*sin_a)/cos_2a, (h*cos_a - w*sin_a)/cos_2a

  return wr,hr

w_r, h_r = rotatedRectWithMaxArea(w,h, np.radians(d) )
h_bb, w_bb, depth = R1.shape

xx = int(np.round((w_bb-w_r)/2))
yy = int(np.round((h_bb-h_r)/2))

R3 = R1[yy:h_bb-yy, xx:w_bb-xx]


# resize all of them to be same ratio
w,h,d = R3.shape


fig, axs = plt.subplots(1,3, figsize=(20,10), gridspec_kw={'wspace': 0.01, 'hspace': 0})
axs[0].imshow(R1)
axs[1].imshow(R2)
axs[2].imshow(R3)
axs[0].set_title('Rotate and expand {}x{}'.format(R1.shape[1],R1.shape[0]))
axs[1].set_title('Rotate as-is {}x{}'.format(R2.shape[1], R2.shape[0]))
axs[2].set_title('Rotate and crop {}x{}'.format(R3.shape[1], R3.shape[0]))
axs[0].axis('off')
axs[1].axis('off')
axs[2].axis('off')
