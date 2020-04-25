#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
from sklearn.utils import Bunch
from sklearn.model_selection import GridSearchCV, train_test_split
import skimage
from skimage.transform import resize
from sklearn import svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
# In[ ]:


def load_image_files(container_path, dimension=(35, 40)):
    
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


# In[ ]:


image_dataset =load_image_files("..\\Labels")


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(
    image_dataset.data, image_dataset.target, test_size=0.3,random_state=109)


# ## SVM

# In[ ]:


param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
 ]
svc = svm.SVC()
clf = GridSearchCV(svc, param_grid)
clf.fit(X_train, y_train)


# In[ ]:


y_pred = clf.predict(X_test)


# In[ ]:


print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))


# In[ ]:


logreg = LogisticRegression(multi_class='multinomial')
logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test)
print("Classification report for - \n{}:\n{}\n".format(
    logreg, metrics.classification_report(y_test, y_pred)))


# ## Guassian NB

# In[ ]:



gaus = GaussianNB()
gaus.fit(X_train, y_train)
y_pred = gaus.predict(X_test)
print("Classification report for - \n{}:\n{}\n".format(
    gaus, metrics.classification_report(y_test, y_pred)))


# ## Random Forest

# In[ ]:


clf = RandomForestClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))


# ## Decision Tree

# In[ ]:


clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Classification report for - \n{}:\n{}\n".format(
    clf, metrics.classification_report(y_test, y_pred)))


# ## XG Boost

# In[ ]:


model=XGBClassifier()
model.fit(X_train,y_train)
y_pred = clf.predict(X_test)
print("Classification report for - \n{}:\n{}\n".format(
    model, metrics.classification_report(y_test, y_pred)))

