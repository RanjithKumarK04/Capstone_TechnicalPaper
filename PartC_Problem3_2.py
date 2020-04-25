#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Import required libraries:

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import joblib
import skimage
from skimage.io import imread
from skimage.transform import resize

# In[8]:


# Import required library
import joblib

# provide local directory path for joblib file
file = "..\\svm_model_joblib"

# load the joblib file into variable model using joblib.load function
model = joblib.load(file)

# provide local directory path for second joblib file
file_1 = "..\\svm_model_joblib_1"

# load the joblib file into variable model_1 using joblib.load function
model_1 = joblib.load(file_1)


# In[9]:


# blank_box fn: this fn is used to detect the empty images uploaded by the user.
# input arguements are: 1. image path and 2. dimensions
# output: if there are any blank_boxes in the cropped images then list of flat data is stored in a variable y_pred
# Once this function find any blank boxes, the main function will break the loop and comes out of the loop.

def blank_box(container_path, dimension=(35,40,3)):
    

    flat_data = []
    flat_data1=[]
    
    # run a for loop for all images to identify the blqnk images and predict the value for that images
    for j in range(0,1):
        
        b = str(j)
        img = imread(container_path + '\\' + b + '.jpg')  # read the image using imread function
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')   # resize the loaded image
        flat_data.append(img_resized.flatten())    # flatten the image into vector format
        flat_data1 = np.array(flat_data)       # convert that vector into array format
        y_pred = model.predict(flat_data1)     # predict the value using the model
    
    return y_pred


# In[10]:


# This function is used to recognize the object in the image using ml model
# Input for this function is 1. Images path and 2. dimension conversion
# output for this funciton is a list of predicted values for the images in a folder

def load_image_test(container_path, dimension=(35,40,3)):
    
    # Initializing three lists and i = 0

    flat_data = []
    predict = []
    flat_data1 = []
    i = 0
    
    # get the number of images from the provided directory
    for file in os.listdir(container_path):
        i += 1
        
    # run a for loop for all images to identify the object in the image and predict the value
    for j in range(i):        
        b=str(j)
        img = imread(container_path + '\\' + b + '.jpg')  # read the image using imread function
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')   # resize the loaded image
        flat_data.append(img_resized.flatten())    # flatten the image into vector format
        flat_data1 = np.array(flat_data)       # convert that vector into array format
        y_pred = model_1.predict(flat_data1)     # predict the value using the model
        
    predict = y_pred
    
    return predict
