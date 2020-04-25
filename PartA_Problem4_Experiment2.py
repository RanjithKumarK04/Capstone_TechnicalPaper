#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import required libraries:

import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2
import statistics


# In[2]:


## Final Code (modified on 26-02-2020)

def angle_image(image):
    
    # import the necessary packages
    import numpy as np
       
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


# In[3]:


## Final Code (modified on 26-02-2020)

def rotation_image(angle,image):
    
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


# In[5]:


## Final Code (modified on 26-02-2020)

# Read the Image into variable "a":
a = cv2.imread("..\\rotated_image_16.jpg")

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

# Save the Image into local directory:
cv2.imwrite("..\\"+"rotated_image_16.jpg", a)


# In[ ]:


## Final Code

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



# In[8]:


a = angle_rotation("..\\rotated_image_16.jpg")


# In[9]:


## Final Code

def contours(input):
    
    import cv2
    
    img_grey = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    
    thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)

    # Find the number of Contours for the edged Image:

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    print("Number of Contours found = " + str(len(contours)))
    
    rectangles = [cv2.boundingRect(count) for count in contours]

    # print("No of contours: ",len(rectangles))

    rectangles_30_30 = []

    for i in rectangles:
        if (i[2] > 30 and i[2] < 60 and i[3] > 37 and i[3] < 60):
            rectangles_30_30.append(i)
            
    return rectangles_30_30


# In[10]:


rectangles_30_30 = contours(a)


# In[11]:


# Calculate the combined bounding rectangle points:

top_x = min([x for (x, y, w, h) in rectangles_30_30])
top_y = min([y for (x, y, w, h) in rectangles_30_30])
bottom_x = max([x+w for (x, y, w, h) in rectangles_30_30])
bottom_y = max([y+h for (x, y, w, h) in rectangles_30_30])

print('top_x: ', top_x, 'top_y: ', top_y)
print('bottom_x: ', bottom_x, 'bottom_y: ', bottom_y)


# In[78]:


file_path = "..\\rotated_image_16.jpg"


import PIL
image = PIL.Image.open(file_path)

for i in range(1,2):
    left = (top_x  + 40 * i)
    upper = top_y
    right = (left + 40)
    bottom = bottom_y
    cropped_image = image.crop((left, upper, right, bottom))
    cropped_image.save("..\\"+"cropped_"+str(i) +".jpg")


# In[91]:


## Working Code:

def contours(input):
    
    import cv2
    import numpy
    
    if type(input) == numpy.ndarray:
        img_grey = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
        print(type(img_grey))
    else:      
        img_grey = cv2.cvtColor(np.array(input), cv2.COLOR_BGR2GRAY)
        print(type(img_grey))
        
    thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)

    # Find the number of Contours for the edged Image:

    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    return contours


# In[93]:


contours = contours(cropped_image)


# In[ ]:   
rectangles = [cv2.boundingRect(count) for count in contours]

    # print("No of contours: ",len(rectangles))

rectangles_30_30 = []

for i in rectangles:
    if (i[2] > 30 and i[2] < 60 and i[3] > 37 and i[3] < 60):
        rectangles_30_30.append(i)


# In[18]:


import cv2

def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return cnts, boundingBoxes


# In[94]:


sorted_con,bounding_boxes = sort_contours(contours,"top-to-bottom")


# In[95]:


bounding_boxes


# In[82]:


rectangles_30_30 = []

for i in bounding_boxes:
        if (i[2] > 30 and i[2] < 60 and i[3] > 37 and i[3] < 60):
            rectangles_30_30.append(i)
print(len(rectangles_30_30))


# In[83]:


rectangles_30_30


# In[84]:


rectangles_30_30_1 = [[i] for i in rectangles_30_30]


# In[85]:


rectangles_30_30_1


# In[36]:



h_mode = statistics.mode([h for (x, y, w, h) in rectangles_30_30])

n = int(((bottom_y-top_y)/h_mode))


# In[53]:


temp=[]

b1 = [[] for i in range(0, n+1)]

for j in range (1, n+1):
    for i in rectangles_30_30:
        x=i[0]
        y=i[1]
        w=i[2]
        h=i[3]            
        if w > 30 and w < 60 and h > 37 and h < 60:
            
            b1[j].append(i)
            temp.append(i)
b1.pop(0)


# In[61]:


b1


# In[86]:


imgray = cv2.imread("..\\cropped_1.jpg",0)


# In[87]:


count=0

for i in rectangles_30_30_1:
    while i != 30:
        len1=len(i)
        for j in i:
            x=j[0]
            y=j[1]
            w=j[2]
            h=j[3]
            if ((w > 30 and w < 60) and (h > 37 and h < 60)):
                cv2.imwrite("..\\"+str(count)+".jpg",imgray[y:y+h,x:x+w])
                count+=1
            break
        break

