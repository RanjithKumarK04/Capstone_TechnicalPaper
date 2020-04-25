#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def empty_columns(excel_path, course_name):
    import pandas as pd
    
    data = pd.read_excel(excel_path, sheet_name = course_name)
    cols = data.columns
    
    list_columns = []

    for i in range(len(data.columns)):
        a = cols[i]
        if data[a].isnull().sum() != 0:
            list_columns.append(i)
    return list_columns