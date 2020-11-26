import cv2
import numpy as np

ROTATE_AND_EXPAND = 1
ROTATE_AND_CROP = 2
ROTATE_AND_KEEP = 0


def rotate_image(image : np.ndarray, 
                 degrees : float = 0, 
                 rotation_type : int = ROTATE_AND_CROP) -> np.ndarray:
    '''
    Rotates image by given angle anti-clockwise with given type of post-processing 

    Parameters
    ----------
    image (OpenCV image) :
        Image imported by OpenCV to rotate
    
    degrees (float): 
        Degrees to rotate image anti-clockwise. For clockwise rotation, provide negative value. The default is 0.
    
    rotation_type (int) :
        Post-processing of rotation. 
        Possible values: 
            ROTATE_AND_KEEP (0) - rotate but keep canvas size. Cuts corners after rotation. 
            ROTATE_AND_EXPAND (1) - rotate and expand canvas size. Expands to fit rotated image. 
            ROTATE_AND_CROP (2) - rotate and crop image to keep only image information and remove black areas from rotation. 
        The default is ROTATE_AND_CROP

    Returns
    -------
    Rotated image 

    '''
    
    # check if image has 'shape'
    if 'shape' not in dir(image):
        raise TypeError("No image provided, or it is of incorrect type")
    # get pictiure dimensions
    elif len(image.shape) == 3:
        # for colour and other 3-dimensional spaces
        height, width, depth = image.shape
    elif len(image.shape) == 2:
        # for greyscale space
        height, width = image.shape
    # raise exception as can't proceed
    else:
        raise ValueError("Provided image is in incorrect format")
        
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
        # in case of expanding and cropping - calculate expanded matrix and new dimensions
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
        
        # calculate new dimensions of fitted image (for cropping)
        if side_short <= 2.0 * sin_t * cos_t * side_long or abs(sin_t - cos_t) < 1e-10:
            # half constrained case: two crop corners touch the longer side,
            # the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            crop_width, crop_height = ( x / sin_t, x / cos_t) if width >= height else ( x / cos_t, x / sin_t)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2t = cos_t * cos_t - sin_t * sin_t
            crop_width, crop_height = (width * cos_t - height * sin_t) / cos_2t, (height * cos_t - width * sin_t) / cos_2t

        # get coordinates of the new dimension
        dx = int(np.round((new_width - crop_width) / 2))
        dy = int(np.round((new_height - crop_height) / 2))
        
        # crop image
        rotated_image = rotated_image[dy : new_height - dy, dx : new_width - dx]
        
    # return rotated image
    return rotated_image