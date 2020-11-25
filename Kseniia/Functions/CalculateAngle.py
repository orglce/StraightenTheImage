import cv2
import numpy as np
from collections import Counter


# default weights for calculating
# w1 - median angle
# w2 - longest line
# w3 - every line
# w4 - parallel lines
# w5 - perpendicular lines
# w6 - flat lines (vertical or horizontal)
DEFAULT_WEIGHTS = [1,2,0.1,1.25,1.5,1]


def calculate_angle(image : np.ndarray, hough_lines : np.ndarray = None, hough_lines_p : np.ndarray = None, weights : list = DEFAULT_WEIGHTS, significance : float = 10 ) -> float:
    '''
    Calculates the most important angle, or angle to which need to rotate image to get it straight

    Parameters
    ----------
    image (OpenCV image) :
        Image imported by OpenCV to calculate dimensions from
    
    hough_lines (OpenCV HoughLine) :
        List of lines provided by OpenCV HoughLine function
    
    hough_lines_p (OpenCV HoughLineP) :
        List of lines provided by OpenCV HoughLineP function
    
    weights (list) :
        List of weights for calculating lines importance:
            w1 - median angle;  
            w2 - longest line;  
            w3 - every line;  
            w4 - parallel lines;    
            w5 - perpendicular lines; 
            w6 - flat (vertical or horizontal line)
        The default is DEFAULT_WEIGHTS.

    significance (float) :
        Define line with length of what percentage of image's diagonal considred to be signifficant. Default is 10%

    Returns
    -------
    The most important angle of image

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
    
    # check if either hough_lines or hough_lines_p was provided
    if hough_lines is None and hough_lines_p is None:
        raise ValueError("Both hough_lines and hough_lines_p can't be empty!")
    
    # check if weights list is in correct format
    if type(weights) != list:
        raise TypeError('Weights list is of incorrect type')
    elif len(weights) != 6:
        raise ValueError('Weights list is in incorrect format')
    
    final_angle = 0
    diagonal = np.round(np.sqrt(height**2 + width**2))
    significance = significance / 100 if significance >= 1 else significance
    
    # unpack weights
    weight_median, weight_length, weight_each, weight_par, weight_perp, weight_flat = weights
    
    # compose all angles from both hough_lines and hough_lines_p into one list of dictionaries
    angles = []
    
    if hough_lines is not None:
        for line in hough_lines:
            # catch exceptions if for some reason format is wrong
            try:
                # get angle (in radians)
                theta = line[0][1]
                # convert to degrees
                theta = np.degrees(theta)
                # add angle to the list with empty length
                angles.append({'angle': theta, 'length': -1 })
            except:
                raise ValueError("hough_lines is in incorrect format")
    
    if hough_lines_p is not None:
        for line in hough_lines_p:
            # catch exceptions if for some reason format is wrong
            try:
                # get line' extreme points
                x1, y1, x2, y2 = line[0]
                # calculate length
                length = np.round( np.sqrt((x2-x1)**2 + (y2-y1)**2), 1)
                # calculate angle
                theta = np.arctan2(y2-y1, x2-x1) # - np.pi/2
                # convert to degrees and shift 90 degrees to match output of hough_lines
                theta = np.round(np.degrees(theta)) - 90
                if theta < 0:
                    theta = theta + 180
                    
                # process only "significant" lines, which are lengthier than "significance" percentage of image's diagonal
                if length >= diagonal * significance:
                    # add angle to the list with empty length
                    angles.append({'angle': theta, 'length': length })
            except:
                raise ValueError("hough_lines_p is in incorrect format")
    

    # sort angles by length ascending order
    angles = sorted(angles, key = lambda angle: angle['length'], reverse=True)
    # separate angles from their length
    just_angles = [element['angle'] for element in angles]
    # sum count lines with same angles
    counted_angles = Counter(just_angles).most_common()
    # get unique angles
    unique_angles = set(just_angles)

    # calculate weighted lines
    weighted_angles = {}
    

    # Add median angle with 'w1' weight
    weighted_angles[np.median(just_angles)] = weighted_angles.get(np.median(just_angles), 0) + weight_median
 
 
    # Add longest line with 'w2' weight (but only if it is longer than 0)
    longest = angles[0]['length'] if angles[0]['length'] > 0 else 0
    for line in angles:
        # get longest line (may be multiple lines with different angles)
        if line['length'] >= longest:
            weighted_angles[line['angle']] = weighted_angles.get(line['angle'], 0) + weight_length
        else:
            # stop processing lines, as it is already shorter than the longest one
            break
    
    
    # Add each line with 'w3' weight
    for angle in just_angles:
        weighted_angles[angle] = weighted_angles.get(angle, 0) + weight_each
    
    
    # Add parallel lines with 'w4' weight
    # get only angles where count > 1 (same angle so parallel lines)
    for line in [(element[0], element[1]) for element in counted_angles if element[1] > 1]:
        weighted_angles[line[0]] = weighted_angles.get(line[0], 0) + line[1] * weight_par
    
    
    # Add perpendicual lines with 'w5' weight
    for angle in just_angles:
        # check if there is a perpendicualr pair (+/- 90) and get lowest of pair
        if (angle + 90) in unique_angles:
            pair_angle = angle
        elif (angle - 90) in unique_angles:
            pair_angle = angle - 90
        else:
            pair_angle = -1

        # if pair found: add as weighted
        if pair_angle != -1:
            weighted_angles[pair_angle] = weighted_angles.get(pair_angle, 0) + weight_perp
            
    
    # Add flat (vertical or horizontal) lines with 'w6' weight
    for angle in just_angles:
        if angle == 90.0 or angle == 0.0 or angle == 180.0:
            weighted_angles[angle] = weighted_angles.get(angle, 0) + weight_flat
    
    
    # calculate the most important angle - the one with most entries in weighted list
    final_angle = max(weighted_angles, key = weighted_angles.get)
    
    # translate angle between 0 and 180 degrees to the most suitable rotation angle
    # 0   <= theta < 45   then theta
    # 45  <= theta < 135  then theta - 90
    # 135 <= theta <= 180 then theta - 180
    # positive number is anti-clockwise, where negative number is clockwise
    
    if final_angle >= 45 and final_angle < 135:
        final_angle = final_angle - 90
    elif final_angle >= 135 and final_angle <= 180:
        final_angle = final_angle - 180
    # else - no change in angle
    
    # return the most important angle
    return final_angle
    
   