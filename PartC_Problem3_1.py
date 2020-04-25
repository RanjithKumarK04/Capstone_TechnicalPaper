#!/usr/bin/env python
# coding: utf-8

# In[4]:


## Import required libraries:

import warnings
warnings.filterwarnings('ignore')
import os
import numpy as np
import joblib
import skimage
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
from skimage.io import imread
from skimage.transform import resize

# In[11]:


# This function is used to convert the image into vector form(flat data)
# Input for the funciton is 1. image_path (local directory path) and 2. dimension (transformation) of image
# dimension (35,40,3) is default values
# output will be flat data of image, target value, class details, and description


from pathlib import Path

def load_image_files(container_path, dimension = (35,40,3)):
    
    # saving the folder_path as pathlib.WindowsPath
    image_dir = Path(container_path)
    
    # this will extract all folders in a specific folder path and stores in a list
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    
    # stroing all folders names in a list categories
    categories = [fo.name for fo in folders]
    
    # Initialize three new list named images, flat_data, and target
    images = []
    flat_data = []
    target = []
    
    # i is number of folders in specific folder
    # direc is the path for the folder
    for i, direc in enumerate(folders):
        
        for file in direc.iterdir():           # file is each image in the specific folder
            img = skimage.io.imread(file)      # reading th image using skimage library
            # resize the image based on dimension provided above
            img_resized = skimage.transform.resize(img, dimension, anti_aliasing = True, mode = 'reflect')
            flat_data.append(img_resized.flatten())     # flatten the image Into vector form
            images.append(img_resized)            # append the flatten data into images list
            target.append(i)                      # store the i value in the target list
    # convert the flatten data into numpy array format       
    flat_data = np.array(flat_data)
    target = np.array(target)       # convert the target into numpy array format 
    images = np.array(images)       # images into numpy array format
    
    return Bunch(data = flat_data, target = target, target_names = categories, images = images)


# In[12]:


file = "..\\svm_model_joblib"

model = joblib.load(file)


# In[ ]:


# This function is used to convert the image into vector form(flat data)
# Input for the funciton is 1. image_path (local directory path) and 2. dimension (transformation) of image
# dimension (35,40,3) is default values
# output will be flat data of image, target value, class details, and description

# Import necessary libraries:

def load_image_files(container_path, dimension = (35,40,3)):
    
    # saving the folder_path as pathlib.WindowsPath
    image_dir = Path(container_path)
    
    # this will extract all folders in a specific folder path and stores in a list
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    
    # stroing all folders names in a list categories
    categories = [fo.name for fo in folders]
    
    # Initialize three new list named images, flat_data, and target
    images = []
    flat_data = []
    target = []
    
    # i is number of folders in specific folder
    # direc is the path for the folder
    for i, direc in enumerate(folders):
        
        for file in direc.iterdir():           # file is each image in the specific folder
            img = skimage.io.imread(file)      # reading th image using skimage library
            # resize the image based on dimension provided above
            img_resized = skimage.transform.resize(img, dimension, anti_aliasing = True, mode = 'reflect')
            flat_data.append(img_resized.flatten())     # flatten the image Into vector form
            images.append(img_resized)            # append the flatten data into images list
            target.append(i)                      # store the i value in the target list
    # convert the flatten data into numpy array format       
    flat_data = np.array(flat_data)
    target = np.array(target)       # convert the target into numpy array format 
    images = np.array(images)       # images into numpy array format
    
    return Bunch(data = flat_data, target = target, target_names = categories, images = images)


# In[22]:


def load_image_files(container_path, dimension=(35,40,3)):
    """
    Load image files with categories as subfolder names 
    which performs like scikit-learn sample dataset
    
    Parameters
    ----------
    container_path : string or unicode
        Path to the main folder holding one subfolder per category
    dimension : tuple
        size to which image are adjusted to
        
    Returns
    -------
    Bunch
    """
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    descr = "A image classification dataset"
    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    
    return Bunch(data=flat_data,
                 target=target,
                 target_names=categories,
                 images=images,
                 DESCR=descr)


# In[23]:


image_dataset =load_image_files("C:\\Users\\Soumya\\Desktop\\Capstone_Project\\Labels2")


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)


# In[25]:


param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)


# In[26]:


def blank_box(container_path, dimension=(35,40,3)):
    
    flat_data = []
    predict=[]
    flat_data1=[]

    for j in range(0,1):
        
        b=str(j)
        img = skimage.io.imread(container_path+'\\'+b+'.jpg')
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten()) 
        flat_data1 = np.array(flat_data)
        y_pred = clf.predict(flat_data1)
    
    return y_pred


# In[14]:


# This function is used to recognize the object in the image using ml model
# Input for this function is 1. Images path and 2. dimension conversion
# output for this funciton is a list of predicted values for the images in a folder

# Import necessary libraries:
def load_image_test(container_path, dimension=(35,40,3)):
    

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
        y_pred = model.predict(flat_data1)     # predict the value using the model
        
    predict = y_pred
    
    return predict