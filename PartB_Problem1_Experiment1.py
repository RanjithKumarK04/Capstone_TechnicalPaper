#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import required libraries
from pathlib import Path
import numpy as np
from sklearn.utils import Bunch
import skimage
from skimage.transform import resize

def load_image_files(container_path, dimension=(35,40)):

    image_dir = Path(container_path)
    folders = [directory for directory in image_dir.iterdir() if directory.is_dir()]
    categories = [fo.name for fo in folders]

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
    
    return Bunch(data=flat_data,target=target,target_names=categories,images=images)