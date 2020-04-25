#!/usr/bin/env python
# coding: utf-8

# In[11]:


# Import required libraries:

import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2
from PIL import Image

os.chdir(r"..\New folder")


# In[105]:


#image = cv2.imread("E:\\Praxis\\Capstone Project\\template - 1\\rotated_images\\25-02-2020\\Image_2_300.jpg", 0)

image = cv2.imread("Image_6.jpg", 0)

gray = cv2.bitwise_not(image)

thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

print(angle)

if angle < -45:
    angle = -(90 + angle)
else:
    angle = -angle
    
print(angle)


# In[103]:


(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle , 1.0)
rotated = cv2.warpAffine(image, M, (w, h),
flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
# show the output image
print("[INFO] angle: {:.3f}".format(angle))
    
# Saving the Image into specified folder:
cv2.imwrite("..\\rotated_Image_2.jpg", rotated)


# In[101]:


image = cv2.imread("..\\rotated_Image_2.jpg", 0)

gray = cv2.bitwise_not(image)

thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

print("angle after rotation: ",angle)


# In[106]:


#image = cv2.imread("E:\\Praxis\Capstone Project\\template - 1\\rotated_images\\25-02-2020\\rotated_image_1.jpg", 0)

image = cv2.imread("Image_6.jpg", 0)

imgray = 255 - image

thresh = cv2.adaptiveThreshold(imgray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,17,2)

# Find the number of Contours for the edged Image:

contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

print("Number of Contours found = " + str(len(contours)))


# In[107]:


# Creating a list of rectangle boxes with width greater than 30 and length greater than 30 from bounding rectangles:

# Calculate bounding rectangles for each contour:

rectangles = [cv2.boundingRect(count) for count in contours]

print("No of contours: ",len(rectangles))

rectangles_30_30 = []

for i in rectangles:
    if (i[2] > 30 and i[2] < 60 and i[3] > 37 and i[3] < 60):
        rectangles_30_30.append(i)
        
print("No of contours which has width and height greater than 30 pixels: ", len(rectangles_30_30))


# In[108]:


# Calculate the combined bounding rectangle points:

top_x = min([x for (x, y, w, h) in rectangles_30_30])
top_y = min([y for (x, y, w, h) in rectangles_30_30])
bottom_x = max([x+w for (x, y, w, h) in rectangles_30_30])
bottom_y = max([y+h for (x, y, w, h) in rectangles_30_30])

print('top_x: ', top_x, 'top_y: ', top_y)
print('bottom_x: ', bottom_x, 'bottom_y: ', bottom_y)


# In[71]:


sorted_rect = sorted(rectangles_30_30, reverse=False)


# In[110]:


image = Image.open("Image_6.jpg")

for i in range(20):
    left = top_x  + 40 * i
    upper = top_y
    right = left + 40
    bottom = bottom_y
    cropped_image = image.crop((left, upper, right, bottom))
    cropped_image.save("cropped_"+str(i) +".jpg")


# In[ ]:


import cv2

a = cv2.imread("..\\new_image_7.jpg")

angle = angle_image(a)
print(angle)

while angle != 0:
    a = rotation_image(a)
    angle = angle_image(a)
    print(angle)

cv2.imwrite("..\\"+"rotated_image_4.jpg", a)