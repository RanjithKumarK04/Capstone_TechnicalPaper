#!/usr/bin/env python
# coding: utf-8

# In[1]:


def angle_image(image):
    
    # import the necessary packages
    import numpy as np
    import cv2
       
    # Convert to gray scale Image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # To Invert the every bit of an array:
    gray = cv2.bitwise_not(gray)

    # Fixed-level thresholding to a multiple-channel array
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # get the coordinates using np.column_stack where the thershold is greater than 0:
    coords = np.column_stack(np.where(thresh > 0))

    # Using cv2.minAreaRect function to get the angle of the Image from the above coords:
    angle = cv2.minAreaRect(coords)[-1]
    
    # If the angle is less than -45 degree then sum the angle with 90 degree else take the negative of the resulted angle:
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    return angle


# In[2]:


def rotation_image(angle,image):
    
    # import the necessary packages
    import cv2
    
    # Check if the Angle of the Image is 0:
    if angle != 0:
        
        # Get the height and width of the Image
        (h, w) = image.shape[:2]
        
        # Get the Center of the Image:
        center = (w // 2, h // 2)
        
        # using cv2.RotationMatrix2D to required rotation the Image using calculated center and angle:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Using Affine function which helps in rotating the image based on m, w, and h (three points)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
    return image


# In[3]:


def angle_rotation(file_path):
    import cv2

    # Read the Image into variable "a":
    a = cv2.imread(file_path)

    # Using angle_image function to get the titled angle of the Image:
    angle = angle_image(a)

    # Print the titled angle:
    print("Angle of the Image is: ",angle)
    
    # Enter while loop if the Image angle is not equal to 0:
    while angle != 0:
    
    # Using rotation_image function to tilt the Image(adjust the angle):
        a = rotation_image(angle,a)
    
    # Checking for the angle of Image after Image is tilted:
        angle = angle_image(a)
        
    return a