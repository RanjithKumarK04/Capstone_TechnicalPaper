#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
import warnings
import cv2
import numpy as np
import numpy
import shutil
warnings.filterwarnings('ignore')
from PIL import Image, ImageEnhance
import PIL


def cropped_images(file_path, course_name):
    
    path = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(path,'excel//AttendenceSheet.xlsx')
    
    # using empty_columns function find the number of empty columns from the attendance
    empty_cols_list = empty_columns(excel_path,course_name)
    
    # split the file_path:
    head,tail = os.path.split(file_path)
    
    # creating an local path using above splitted path to save the rotated image
    image_path = head+'\\'+'rotated_image.jpg'

    # Checking for if an Image is tilted, If titled, below function will adjust the tilt. 
    a = angle_rotation(file_path)
    
    # save the rotated image in the local directory(location is mentioned below)
    cv2.imwrite(image_path, a)

    # Finding the Initial contours for the Image:
    contours_initial = contours(image_path,0)

    # Calculate the combined bounding rectangle points:
    top_x = min([x for (x, y, w, h) in contours_initial])
    top_y = min([y for (x, y, w, h) in contours_initial])
    bottom_x = max([x+w for (x, y, w, h) in contours_initial])
    bottom_y = max([y+h for (x, y, w, h) in contours_initial])
    
    # running a for loop:(where i is list of empty_cols_list)
    for i in empty_cols_list:
        
        j = i - 2
        
        # storing i as the column nnumber
        col_number = i
            
        # Creating a new Folder:
        os.mkdir(head+"\\"+str(j))
            
        # Load the Image:
        image = PIL.Image.open(image_path)
            
        # Get the co-ordinates for cropping the required Image:
        left = (top_x  + 40 * j)
        upper = top_y
        right = (left + 40)
        bottom = bottom_y
            
        # Crop the Image using above values:
        cropped_image = image.crop((left, upper, right, bottom))
            
        # Enchance the contrast of the cropped_image:
        enhancer = ImageEnhance.Contrast(cropped_image)
        enhanced_im = enhancer.enhance(40)
            
        # Convert the image into grayscale image:
        imgrey = numpy.array(enhanced_im)

        # Get the contours for the cropped Image using creatd contours function:
        contours_cropped_image = contours(imgrey,1)
        
        # sorting the contours using sort_contours function
        sorted_con,bounding_boxes = sort_contours(contours_cropped_image,"top-to-bottom")

        list_bounding_boxes = [[a] for a in bounding_boxes]
            
        # Changing the directory to save the cropped Images:
        os.chdir(head+"\\"+str(j))
        
        # changing the directory
        images_dir = os.getcwd()
        
        # Initializing num = 0:
        num = 0
                
        # Saving the cropped Images into above specified library:
        for b in list_bounding_boxes:
            while b != 30:
                len1=len(b)
                for j in b:
                    x=j[0]
                    y=j[1]
                    w=j[2]
                    h=j[3]
                    if ((w > 30 and w < 60) and (h > 35 and h < 60)):
                        cv2.imwrite(str(num)+".jpg",imgrey[y:y+h,x:x+w])
                        num += 1
                    break
                break
                
        # recognization the blank box:
        Blank_Boxes = blank_box(images_dir)
        
        if Blank_Boxes == 0:
            # recognization and predicting the values:
            predicted_values = load_image_test(images_dir)
        
            # convert the predicted values into list
            predicted_values_list = predicted_values.tolist()
        
            # update the attendance excel sheet wiht above predicted value list in the specified column number
            update_attendance(excel_path, predicted_values_list, course_name, col_number)
            
            # delete the created folders in the local directory
            shutil.rmtree(images_dir)
        else:
            # delete the created folders in the local directory
            shutil.rmtree(images_dir)
            print("Stopped processing as blank boxes were found in the Image")
            break
    img_path = os.path.join(path,image_path)
    os.remove(img_path)           
    return "successfully_saved_images"

