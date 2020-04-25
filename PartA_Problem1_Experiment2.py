#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import required libraries:

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2
from PIL import Image

os.chdir("..\\Rework")


# In[2]:


# Import the required Image into python notebook:

image = cv2.imread("new_image_7.jpg", 0)


# In[3]:


## Rotating the Image:

gray = cv2.bitwise_not(image)

thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

coords = np.column_stack(np.where(thresh > 0))
rRect = cv2.minAreaRect(coords)

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
    
print(angle)

(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# show the output image
print("[INFO] angle: {:.3f}".format(angle))

## Saving the Image into specified folder:
cv2.imwrite("..\\rotated_image.jpg", rotated)


# In[3]:


# Binarize the image and call it thresh.

image = cv2.imread("rotated_image.jpg", 0)

imgray = 255 - image

thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

# Find the number of Contours for the edged Image:

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of Contours found = " + str(len(contours)))


# In[4]:


#Calculate bounding rectangles for each contour:

rectangles = [cv2.boundingRect(count) for count in contours]

print("No of contours: ",len(rectangles))

#print(rectangles.pop(0))


# In[8]:


# Creating a list of rectangle boxes with width greater than  30 and length greater than 30 from bounding rectangles:

rectangles_30_30 = []

for i in rectangles:
    if (i[2] > 33 and i[2] < 60 and i[3] > 32 and i[3] <60):
        rectangles_30_30.append(i)
        
print("No of contours which has width and height greater than 30 pixels: ", len(rectangles_30_30))


# In[9]:


#Calculate the combined bounding rectangle points:

top_x = min([x for (x, y, w, h) in rectangles_30_30])
top_y = min([y for (x, y, w, h) in rectangles_30_30])
bottom_x = max([x+w for (x, y, w, h) in rectangles_30_30])
bottom_y = max([y+h for (x, y, w, h) in rectangles_30_30])

print('top_x: ', top_x, 'top_y: ', top_y)
print('bottom_x: ', bottom_x, 'bottom_y: ', bottom_y)


# In[ ]:


image = Image.open("rotated_image.jpg")

# Setting the points for cropped image:

left = b2[0][0][0] - 5
top = b2[0][0][1] - 5
right = ((b2[0][0][0]) + 40 * 20 ) + 12
bottom = ((b2[0][0][1]) + 47 * 30 ) + 8
  
# Cropped image of above dimension 
# (It will not change orginal image) 
cropped_image = image.crop((left, top, right, bottom)) 
  
# Shows the image in image viewer 
#cropped_image.show()

cropped_image.save("cropped_image_1.jpg")

