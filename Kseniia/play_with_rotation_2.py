import cv2
import numpy as np
from matplotlib import pyplot as plt


ROTATE_AND_EXPAND = 1
ROTATE_AND_CROP = 2
ROTATE_AND_KEEP = 0

toProcess = "../sample_photos/simple_shape_rectangle.png"

image = cv2.imread(toProcess)
degrees = 30
rotation_type = ROTATE_AND_CROP

# get pictiure dimentions
if len(image.shape) == 3:
    # for colour and other 3-dimentional spaces
    height, width, depth = image.shape
elif len(image.shape) == 2:
    # for greyscale space
    height, width = image.shape
else:
    # raise exception as can't proceed
    raise TypeError("Provided image is in incorrect format")
    
# get center of rotation/image
cx = width // 2
cy = height // 2
# do not scale, keep as is
scale = 1

# Prepare rotation matrix
M = cv2.getRotationMatrix2D(center = (cx,cy), angle = degrees, scale = scale)
new_width = width * scale
new_height = height * scale    

if rotation_type == ROTATE_AND_EXPAND or rotation_type == ROTATE_AND_CROP:
    # in case of expanding and cropping - calculate expanded matrix and new dimentions
    '''
        Details taken from https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''
    # rotate expanding bounds of image to not cut it off
    # cos and sin of the rotation's angle can be taken from rotation matrix
    cos_t = np.abs(M[0,0])
    sin_t = np.abs(M[0,1])

    # get new width and height using polar to cartesian translation 
    new_width = int(height * sin_t + width * cos_t)
    new_height = int(height * cos_t + width * sin_t)

    # Adjust matrix to move rotated image and make sure any part is not outside
    M[0,2] = M[0,2] + (new_width / 2) - cx
    M[1,2] = M[1,2] + (new_height / 2) - cy


# do actual rotation (and expansion if needed)
rotated_image = cv2.warpAffine(image, M = M, dsize = (new_width, new_height))

if rotation_type == ROTATE_AND_CROP:
    # in case of cropping - calculate size of the most effective area fitting into rotated image
    '''
        Details taken from https://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
    '''
    # get long and short side
    side_long, side_short = (width, height) if width >= height else (height, width)
    
    # calculate new dimentions of fitted image (for cropping)
    if side_short <= 2.0 * sin_t * cos_t * side_long or abs(sin_t - cos_t) < 1e-10:
        # half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = 0.5 * side_short
        crop_width, crop_height = ( x / sin_t, x / cos_t) if width >= height else ( x / cos_t, x / sin_t)
    else:
        # fully constrained case: crop touches all 4 sides
        cos_2t = cos_t * cos_t - sin_t * sin_t
        crop_width, crop_height = (width * cos_t - height * sin_t) / cos_2t, (height * cos_t - width * sin_t) / cos_2t

    # get coordinates of the new dimention
    dx = int(np.round((new_width - crop_width)/2))
    dy = int(np.round((new_height - crop_height)/2))
    
    R1 = rotated_image.copy()
    cv2.rectangle(R1, pt1=(dx,dy), pt2=(new_width - dx, new_height - dy), color=(255,0,255), thickness = 2)
    
    # crop image
    rotated_image = rotated_image[dy : new_height - dy, dx : new_width - dx]
    R2 = rotated_image.copy()
    
    
    fig, axs = plt.subplots(1,2, figsize=(20,10), gridspec_kw={'wspace': 0.01, 'hspace': 0})
    axs[0].imshow(R1)
    axs[1].imshow(R2)
    axs[0].set_title('Rotate and expand {}x{}'.format(R1.shape[1],R1.shape[0]))
    axs[1].set_title('Rotate and crop {}x{}'.format(R2.shape[1], R2.shape[0]))
    axs[0].axis('off')
    axs[1].axis('off')

