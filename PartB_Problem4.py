#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import numpy as np
from sklearn.utils import Bunch
import os
import skimage
import joblib
from skimage.io import imread
from skimage.transform import resize

# In[ ]:


def load_image_files(container_path, dimension=(35,40,3)):
    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

    images = []
    flat_data = []
    target = []
    for i, direc in enumerate(folders):
        for file in direc.iterdir():
            img = skimage.io.imread(file)
            img_resized = skimage.transform.resize(img, dimension, anti_aliasing=True, mode='reflect')
            flat_data.append(img_resized.flatten()) 
            images.append(img_resized)
            target.append(i)
    flat_data = np.array(flat_data)
    target = np.array(target)
    images = np.array(images)
    
    return Bunch(data=flat_data,target=target,target_names=categories,images=images,)
# In[ ]:


file = "..\\svm_model_joblib"

model = joblib.load(file)


# ## Testing on Test data

# In[ ]:


def load_image_test(container_path, dimension=(35,40,3)):

    flat_data = []
    predict=[]
    flat_data1=[]
    
    i = 0

    for file in os.listdir(container_path):
        i+= 1
    
    for j in range(i):
        
        b=str(j)
        img = imread(container_path + '\\' + b + '.jpg')
        img_resized = resize(img, dimension, anti_aliasing=True, mode='reflect')
        flat_data.append(img_resized.flatten()) 
        flat_data1 = np.array(flat_data)
        y_pred = model.predict(flat_data1)
        
    predict = y_pred
    
    return predict

