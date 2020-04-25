#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Final Code (modified on 26-02-2020)

## We are using below function to get the tilted angle of the Image:
# Input arguement will be an Image:
# Output will be *Tilted Angle of the Image* which is used in further modification:

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


# In[ ]:


## Final Code (modified on 26-02-2020)
# Below function is used to rotate the Image with specified angle from the above function.
# Input arguments will be 1. Angle & 2. Image:
# Output will be Rotated Image with Angle 0(No tilt in the Image):

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


# In[ ]:


## Final Code (modified on 27-02-2020)
## By Combining above fn's we have created an function which will perform rotation of the Image until we get required result:
# Input argument is *Image location*:
# Output will be *Image with 0 degree tilt*
## This function can handle any tilted Image.

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


# In[ ]:


## Below Function will help to get the contours of the Image and we have used *findContours* function:
# input will be an Image:
# If i = 0, output will be bounding rectangles which are formed from the Image using *boundingRect*
# Else output will be only contours

def contours(input,i):
    
    import cv2
    import numpy as np
    import numpy
    
    # Based on the input of the Image we are converting the Image into grayscale:
    if type(input) == numpy.ndarray:
        img_grey = cv2.cvtColor(input, cv2.COLOR_BGR2GRAY)
    else:      
        img_grey = cv2.cvtColor(np.array(input), cv2.COLOR_BGR2GRAY)
    
    # Using *adaptiveThreshold* function from cv2 library we are getting the threshold of the Image:
    thresh = cv2.adaptiveThreshold(img_grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,15,2)

    # Find the number of Contours for the edged Image:
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if i == 0:
        # Extracting bounding rectangle from the Image using *cv2.boundingRect* function:
        rectangles = [cv2.boundingRect(count) for count in contours]

        rectangles_30_30 = []

        for i in rectangles:
            # Picking rectangles whose width is in between 30 and 60 and height between 37 and 60
            if (i[2] > 30 and i[2] < 60 and i[3] > 37 and i[3] < 60):
                rectangles_30_30.append(i)
        # Printing number of rectangular images found from the above function:
        print("Number of rectangle_boxes found = " + str(len(rectangles_30_30)))

        return rectangles_30_30
    else:
        return contours


# In[ ]:


# From the contours fn, we will get contours of the Image but those contours are generated randomly.
# So we need to sort the contours.
## Below function will help us to sort the contours.
# Input arguement will be above extracted contours and method which will be default *top-to-bottom*
# Output will be sorted contours:

def sort_contours(cnts, method="top-to-bottom"):
    import cv2
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



def cropped_images(file_path,start,end):
    
    # Check for input is number or not:
    if start < 20 and type(start) == int and start < end:
        if end < 20 and type(end) == int and end > start:
            
            # Import Required Libraries:
            import os
            import warnings
            import cv2
            import numpy as np
            warnings.filterwarnings('ignore')
            from PIL import Image, ImageEnhance

            # Checking for If an Image is tilted, If titled, below function will adjust the tilt. 
            a = angle_rotation(file_path)

            # Finding the Initial contours for the Image:
            contours_initial = contours(a,0)

            # Calculate the combined bounding rectangle points:
            top_x = min([x for (x, y, w, h) in contours_initial])
            top_y = min([y for (x, y, w, h) in contours_initial])
            bottom_y = max([y+h for (x, y, w, h) in contours_initial])
            
            # getting the path to create new folder and save the cropped Images:
            head,tail = os.path.split(file_path) 

            #Load the Image:
            image = Image.open(file_path)

            for i in range(start,end):

                # Saving "i" value in a variable to create a folder in the speicifed directory:
                count = i

                # Creating a new Folder:
                os.mkdir(head+"\\"+str(count))

                #Load the Image:
                image = Image.open(file_path)

                # Get the co-ordinates for cropping the required Image:
                left = (top_x  + 40 * i)
                upper = top_y
                right = (left + 40)
                bottom = bottom_y

                # Crop the Image using above values:
                cropped_image = image.crop((left, upper, right, bottom))

                # Enchance the contrast of the cropped_image:
                enhancer = ImageEnhance.Contrast(cropped_image)
                enhanced_im = enhancer.enhance(40)

                # Convert the image into grayscale image:
                imgray = cv2.cvtColor(np.array(enhanced_im), cv2.COLOR_BGR2GRAY)

                # Get the contours for the cropped Image using creatd contours function:
                contours_cropped_image = contours(cropped_image,1)

                sorted_con,bounding_boxes = sort_contours(contours_cropped_image,"top-to-bottom")
                # print(bounding_boxes)

                list_bounding_boxes = [[i] for i in bounding_boxes]
                # print(list_bounding_boxes)
                
                # Changing the directory to save the cropped Images:
                os.chdir(head+"\\"+str(count))

                num = 0
                
                # Saving the cropped Images into above specified library:
                for i in list_bounding_boxes:
                    while i != 30:
                        for j in i:
                            x=j[0]
                            y=j[1]
                            w=j[2]
                            h=j[3]
                            if ((w > 30 and w < 60) and (h > 37 and h < 60)):
                                cv2.imwrite(str(num)+".jpg",imgray[y:y+h,x:x+w])
                                num += 1
                            break
                        break
        else:
            print("Error: End argument should be an interger and end should be greater than start always")
    else:
        print("Error: start arguement should be an interger and start should be less than end always")



## Applying the above function to get cropped Images from an Image loaded from the local directory:

cropped_images("..\\rotated_image_16.jpg",0,20)