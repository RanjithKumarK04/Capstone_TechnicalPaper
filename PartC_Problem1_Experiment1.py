#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pandas as pd


# In[2]:


scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']


# In[6]:


import os
os.chdir(r"C:\Users\Himajak\Desktop")


# In[7]:


credentials = ServiceAccountCredentials.from_json_keyfile_name('capstone-project-269906-769efe9e8727.json', scope)


# In[8]:


gc = gspread.authorize(credentials)


# In[ ]:


wks = gc.open("D 19 - Attendance Tracker").sheet1


# In[30]:


data = wks.get_all_values()
headers = data.pop(1)

df = pd.DataFrame(data, columns=headers)
print(df.head())


# In[24]:


df.to_csv("data.csv")


# In[32]:


headers

