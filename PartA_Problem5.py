#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
# In[3]:


import joblib

file = "E:\\Praxis\\Capstone Project\\03-04-2020\\svm_model_joblib"

model = joblib.load(file)


# In[4]:


def load_image_test(container_path, dimension=(35,40,3)):
    
    flat_data = []
    predict=[]
    flat_data1=[]
    
    i = 0
    import os
    from skimage.io import imread
    from skimage.transform import resize

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



selection = str(input("Select *0* for single file or *1* for multiple files: "))

if selection == '0':
    import os
    import time

    batch = input("Select required batch either Jan_batach or July_batch: ")
    time.sleep(2)
    term = input("Select required Term from the below terms to update attendance: ")
    time.sleep(2)
    course_name = input("Select required course name from the term selected: ")
    time.sleep(2)

    course_path = "..\\course"+"\\"+str(batch)+"\\"+str(term)+"\\"+str(course_name)
    print(course_path)

    files_path = [os.path.abspath(x) for x in os.listdir(course_path)]

    file_dir = []

    for i in files_path:
        head,tail = os.path.split(i)
        image_path = course_path + "\\" + tail
        file_dir.append(image_path)
        time.sleep(2)
        
    for i in file_dir:
        cropped_images(i,course_name)   # Main Code - For recognition and updating excel sheet
        time.sleep(2)
else:
    import time
    import os
    
    folder_path = input("Select the term required to update the all courses attendance: ")
    head,tail = os.path.split(folder_path)
    print(tail)
    
    if tail == 'Term_1':
        term_1_courses = ['APST','BSF','Linear Algebra','Machine Learning - 1','Marketing Research',
                          'Python','RDWH','Statistics - 1']
        print(term_1_courses)
        for i in term_1_courses:
            print(i)
            course_name = i
            course_path = folder_path + "\\" + i

            files_path = [os.path.abspath(x) for x in os.listdir(course_path)]
            time.sleep(2)

            print(files_path)

            file_dir = []

            for j in files_path:
                head,tail = os.path.split(j)
                image_path = course_path + "\\" + tail
                file_dir.append(image_path)
            print(file_dir)

            time.sleep(2)
            for n in file_dir:
                cropped_images(n,course_name)  # Main Code - For recognition and updating excel sheet
                time.sleep(15)
    elif tail == 'Term_2':
        term_2_courses = ['BDS','Econometrics','FINA','I2R','Machine Learning - 2','Marketing Analytics',
                          'Model Interpretation','Statistics - 2']
        for i in term_2_courses:
            course_name = i
            course_path = folder_path + "\\" + i
            
            files_path = [os.path.abspath(x) for x in os.listdir(course_path)]
            time.sleep(2)
            
            file_dir = []

            for j in files_path:
                head,tail = os.path.split(j)
                image_path = course_path + "\\" + tail
                file_dir.append(image_path)
            print(file_dir)
            
            time.sleep(2)
            for n in file_dir:
                cropped_images(n,course_name)  # Main Code - For recognition and updating excel sheet
                time.sleep(15)
    elif tail == 'Term_3':
        term_3_courses = ['Data Engineering','Deep Learning','FDM','HEA','HRA','iSAS','No SQL','OT',
                          'Telecom Analytics','Text Analytics','Web Analytics']
        for i in term_3_courses:
            course_name = i
            course_path = folder_path + "\\" + i
            
            files_path = [os.path.abspath(x) for x in os.listdir(course_path)]
            time.sleep(2)
            
            file_dir = []

            for j in files_path:
                head,tail = os.path.split(j)
                image_path = course_path + "\\" + tail
                file_dir.append(image_path)
            print(file_dir)
            
            time.sleep(2)
            for n in file_dir:
                cropped_images(n,course_name)  # Main Code - For recognition and updating excel sheet
                time.sleep(15)