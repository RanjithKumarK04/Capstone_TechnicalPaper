#!/usr/bin/env python
# coding: utf-8

# In[1]:


from skimage import io
from skimage import color
import skimage


# In[4]:


import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
os.chdir(r"..\Capstone Project")


# In[5]:


image = color.rgb2gray(io.imread("Scan0001.jpg"))


# In[8]:


io.imshow(image)


# In[9]:


image1 = io.imread("Scan0001.jpg")
io.imshow(image1)


# In[10]:


import cv2
import numpy as np
os.chdir(r"..\resized_images")

large = cv2.imread('1.jpg')
small = cv2.cvtColor(large, cv2.COLOR_BGR2GRAY)


# In[11]:


small


# In[14]:


os.chdir(r"..\Capstone Project")
image2 = io.imread("attendence.jpg")
io.imshow(image2)


# In[15]:


# Importing Image class from PIL module 
from PIL import Image
  
# Opens a image in RGB mode 
path = r'..\attendence.jpg'
im = Image.open(path)
  
# Setting the points for cropped image 
left = 1
top = 1
right = 36
bottom = 35
  
# Cropped image of above dimension 
# (It will not change orginal image) 
im1 = im.crop((left, top, right, bottom)) 
  
# Shows the image in image viewer 
im1.show()


# In[7]:


import PIL


# In[2]:


from PIL import Image
import PIL

left = 12
top = 9
right = 43
bottom  = 39

import os
os.chdir("..\cropped images")
im = Image.open('Scan0003.jpg')

for i in range(16):
    for l in range(25):
        im1 = im.crop((left, top, right, bottom))
        #im1.show()
        im1.save('image-'+str(i)+str(l)+'.jpg')
    
        top += 47
        bottom += 47
    
    right += 37
    left += 37
    top = 9
    bottom = 39


# In[ ]:


left = 547
top = 136
right = 573
bottom  = 160


# In[7]:


from PIL import Image
import PIL

left = 547
top = 136
right = 573
bottom  = 160

import os
os.chdir("..\cropped images")
im = Image.open('image.jpg')

for i in range(2):
    for l in range(4):
        im1 = im.crop((left, top, right, bottom))
        #im1.show()
        im1.save('image-'+str(i)+str(l)+'.jpg')
    
        top += 47
        bottom += 47
    
    right += 37
    left += 37
    top = 136
    bottom = 160


# In[5]:


from PIL import Image
import PIL

left = 710
top = 434
right = 742
bottom  = 463

import os
os.chdir("..\cropped images")
im = Image.open('Scan0001.jpg')

for i in range(5):
    for l in range(33):
        im1 = im.crop((left, top, right, bottom))
        #im1.show()
        im1.save('image-'+str(i)+str(l)+'.jpg')
    
        top += 47
        bottom += 47
    
    right += 37
    left += 37
    top = 434
    bottom = 463


# In[19]:


from PIL import Image

left = 8
top = 5
right = 47
bottom  = 43

path = r'..\Scan0003.jpg'
im = Image.open(path)

for i in range(26):
    im1 = im.crop((left, top, right, bottom))
    im1.show()
    
    top += 47
    bottom += 47


# In[15]:


print('image-'+str(2)+str(1)+'.jpg')


# In[13]:


from PIL import Image
import PIL

top = 346
bottom  = 376
right = 709
left = 679

import os
os.chdir("..\cropped images")
im = Image.open('210001.jpg')

for l in range(1):
    im1 = im.crop((left, top, right, bottom))
    im1.show()
    im1.save('image-'+str(l)+'.jpg')
    
    top += 44
    bottom += 44


# In[2]:


from PIL import Image
from PIL import ImageEnhance
import PIL
import os

top = 
bottom  = 
right = 
left = 

os.chdir("..\cropped images")
im = Image.open('new_image_1.jpg')

for i in range(1):
    for l in range(10):
        im1 = im.crop((left, top, right, bottom))
        enhancer = ImageEnhance.Contrast(im1)
        enhanced_im = enhancer.enhance(40)
        enhanced_im.save(str(i)+str(l)+'.jpg')
    
        top += 
        bottom += 
    
    right += 
    left += 
    top = 
    bottom = 


# In[10]:


def image_generator(top1,bottom1,left,right,columns,rows,image_path,save_path):
    from PIL import Image
    from PIL import ImageEnhance
    import PIL
    import os
    
    image = Image.open(image_path)
    
    for i in range(columns):
        top = top1
        bottom = bottom1
            
        for l in range(rows):
            image1 = image.crop((left, top, right, bottom))
            enhancer = ImageEnhance.Contrast(image1)
            enhanced_image = enhancer.enhance(40)
            os.chdir(save_path)
            enhanced_image.save(str(i)+str(l)+'.jpg')
    
            top += 48.5
            bottom += 48.5
    
    right += 38.5
    left += 38.5
    return("Successfully saved files in the specified folder")


# In[8]:


image_generator(137,160,548,573,2,10,r"..\image.jpg",r"..\cropped images")


# In[14]:


## CP_GN_1:

image_generator(431,461,723,755,2,20,r"..\CP_GN_1.jpg",r"..\1")


# In[4]:


import cv2
image = cv2.imread(r"..\00.jpg", 0)
image2 = 255 - image
ret, thresh = cv2.threshold(image2, 100, 255, cv2.THRESH_BINARY)
cv2.imwrite("img.jpg",thresh)


# In[ ]:


# Enhancing the cropped images and saving them
from PIL import Image
from PIL import ImageEnhance
import PIL
import os

left = 12
top = 9
right = 43
bottom  = 39


os.chdir(r'..\cropped_images')
im = Image.open('Scan0003.jpg')
enh_path = "..\\enhanced"

for i in range(16):
    for l in range(25):
        im1 = im.crop((left, top, right, bottom))
        #im1.show()
        im1.save('image-'+str(i)+str(l)+'.jpg')
        name = 'image-'+str(i)+str(l)+'.jpg'
        im1 = Image.open(name)
        enhancer = ImageEnhance.Contrast(im1)
        enhanced_im = enhancer.enhance(50)
        enhanced_im.save('enhanced_image-'+str(i)+str(l)+'.jpg')
    
        top += 47
        bottom += 47
    
    right += 37
    left += 37
    top = 9
    bottom = 39
    
    #im1.show()
        #im1.save('image-'+str(i)+str(l)+'.jpg')


# In[1]:


# Python code to find the co-ordinates of 
# the contours detected in an image. 
import numpy as np
import cv2


# In[8]:


# Reading image 
font = cv2.FONT_HERSHEY_COMPLEX

import os
os.chdir(r"..\Capstone Project")
img2 = cv2.imread('new_image_4.jpg', cv2.IMREAD_COLOR)


# In[9]:


# Reading same image in another  
# variable and converting to gray scale. 
img = cv2.imread('new_image_4.jpg', cv2.IMREAD_GRAYSCALE)


# In[10]:


# Converting image to a binary image 
# ( black and white only image).

_, threshold = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY)


# In[11]:


# Detecting contours in image.

contours, _= cv2.findContours(threshold, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)


# In[12]:


# Going through every contours found in the image. 
for cnt in contours : 
  
    approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
  
    # draws boundary of contours. 
    cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)  
  
    # Used to flatted the array containing 
    # the co-ordinates of the vertices. 
    n = approx.ravel()  
    i = 0
  
    for j in n : 
        if(i % 2 == 0): 
            x = n[i] 
            y = n[i + 1] 
  
            # String containing the co-ordinates. 
            string = str(x) + " " + str(y)  
  
            if(i == 0): 
                # text on topmost co-ordinate. 
                cv2.putText(img2, "Arrow tip", (x, y), 
                                font, 0.5, (255, 0, 0))  
            else: 
                # text on remaining co-ordinates. 
                cv2.putText(img2, string, (x, y),  
                          font, 0.5, (0, 255, 0))  
        i = i + 1
  
# Showing the final image. 
cv2.imshow('image2', img2)  
  
# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'):  
    cv2.destroyAllWindows()


# In[20]:


cnt = contours[1]


# In[24]:


cnt


# In[22]:


approx = cv2.approxPolyDP(cnt, 0.009 * cv2.arcLength(cnt, True), True) 
  
# draws boundary of contours. 
cv2.drawContours(img2, [approx], 0, (0, 0, 255), 5)


# In[23]:


approx


# In[26]:


n = approx.ravel()


# In[28]:


i = 0

for j in n:
    if(i % 2 == 0):
        x = n[i] 
        y = n[i + 1]
        
        string = str(x) + " " + str(y)  
  
    if(i == 0):
        # text on topmost co-ordinate. 
        cv2.putText(img2, "Arrow tip", (x, y),font, 0.5, (255, 0, 0))  
    else:
        # text on remaining co-ordinates.
        cv2.putText(img2, string, (x, y),font, 0.5, (0, 255, 0))  
        i = i + 1
  
# Showing the final image. 
cv2.imshow('image2', img2)  
  
# Exiting the window if 'q' is pressed on the keyboard. 
if cv2.waitKey(0) & 0xFF == ord('q'):  
    cv2.destroyAllWindows()


# In[29]:


# import the necessary packages
import imutils
import cv2
# load the image, convert it to grayscale, and blur it slightly
image = cv2.imread("new_image_4.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
# threshold the image, then perform a series of erosions +
# dilations to remove any small regions of noise
thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=2)
# find contours in thresholded image, then grab the largest
# one
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
c = max(cnts, key=cv2.contourArea)


# In[30]:


# determine the most extreme points along the contour
extLeft = tuple(c[c[:, :, 0].argmin()][0])
extRight = tuple(c[c[:, :, 0].argmax()][0])
extTop = tuple(c[c[:, :, 1].argmin()][0])
extBot = tuple(c[c[:, :, 1].argmax()][0])


# In[31]:


# draw the outline of the object, then draw each of the
# extreme points, where the left-most is red, right-most
# is green, top-most is blue, and bottom-most is teal
cv2.drawContours(image, [c], -1, (0, 255, 255), 2)
cv2.circle(image, extLeft, 8, (0, 0, 255), -1)
cv2.circle(image, extRight, 8, (0, 255, 0), -1)
cv2.circle(image, extTop, 8, (255, 0, 0), -1)
cv2.circle(image, extBot, 8, (255, 255, 0), -1)
# show the output image
cv2.imshow("Image", image)
cv2.waitKey(0)


# In[2]:


import cv2
import numpy as np
import os
os.chdir(r"..\Capstone Project")

img = cv2.imread('new_image_4.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bilateralFilter(gray, 11, 17, 17)


# In[3]:


kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 2)
kernel = np.ones((4,4),np.uint8)
dilation = cv2.dilate(erosion,kernel,iterations = 2)

edged = cv2.Canny(dilation, 30, 200)


# In[5]:


edged.


# In[6]:


contours, hierarchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

rects = [cv2.boundingRect(cnt) for cnt in contours]
rects = sorted(rects,key=lambda  x:x[1],reverse=True)


# In[11]:


fileName = ['9','8','7','6','5','4','3','2','1','0']

i = -1
j = 1
y_old = 5000
x_old = 5000
for rect in rects:
    x,y,w,h = rect
    area = w * h

    if area > 5 and area < 20:

        if (y_old - y) > 20:
            i += 1
            y_old = y

        if abs(x_old - x) > 20:
            x_old = x
            x,y,w,h = rect

            out = img[y+5:y+h-5,x+5:x+w-5]
            cv2.imwrite('cropped\\' + fileName[i] + '_' + str(j) + '.jpg', out)

            j+=1


# In[13]:


import cv2;
import numpy as np;

# Run the code with the image name, keep pressing space bar

# Change the kernel, iterations, Contour Area, position accordingly
# These values work for your present image

img = cv2.imread("new_image_4.jpg", 0);
h, w = img.shape[:2]
kernel = np.ones((15,15),np.uint8)

e = cv2.erode(img,kernel,iterations = 2)  
d = cv2.dilate(e,kernel,iterations = 1)
ret, th = cv2.threshold(d, 150, 255, cv2.THRESH_BINARY_INV)

mask = np.zeros((h+2, w+2), np.uint8)
cv2.floodFill(th, mask, (200,200), 255); # position = (200,200)
out = cv2.bitwise_not(th)
out= cv2.dilate(out,kernel,iterations = 3)
cnt, h = cv2.findContours(out,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for i in range(len(cnt)):
            area = cv2.contourArea(cnt[i])
            if(area>10000 and area<100000):
                  mask = np.zeros_like(img)
                  cv2.drawContours(mask, cnt, i, 255, -1)
                  x,y,w,h = cv2.boundingRect(cnt[i])
                  crop= img[ y:h+y,x:w+x]
                  cv2.imshow("snip",crop )
                  if(cv2.waitKey(0))==27:break

cv2.destroyAllWindows()

