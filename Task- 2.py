#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn .model_selection import train_test_split
from sklearn import metrics
from xgboost import XGBRegressor
import warnings
warnings.simplefilter("ignore")


# # Load Dataset

# In[2]:


df=pd.read_csv("sales.csv")


# In[3]:


df


# In[4]:


df.describe()


# In[5]:


df.info()


# In[6]:


ser=pd.Series(np.random.rand)


# In[7]:


type(ser)


# In[8]:


df.dtypes


# # Finding Missing Values

# In[9]:


df.isnull().sum() 


# In[10]:


mode_of_outlet_size = df.pivot_table(values='Outlet_Size',columns='Outlet_Type',aggfunc=(lambda x:x.mode()[0])) 
  


# In[11]:


print(mode_of_outlet_size)


# In[12]:


missing_values=df['Outlet_Size'].isnull()


# In[13]:


print(missing_values)


# In[14]:


df.loc[missing_values,'Outlet_Size']=df.loc[missing_values,'Outlet_Type'].apply(lambda x:mode_of_outlet_size)


# In[15]:


df.isnull().sum()


# In[16]:


mode_of_Item_weight = df.pivot_table(values='Item_Weight',columns='Item_Type',aggfunc=(lambda x:x.mode()[0])) 
  


# In[17]:


print(mode_of_Item_weight)


# # Feature Selection

# In[18]:


x=df.iloc[:,:3]
y=df.iloc[:,-1]


# In[19]:


x.head()


# In[20]:


y.head()


# # Analysis of Data

# In[21]:


df.describe()


# In[22]:


sns.set()


# In[23]:


plt.figure(figsize=(6,6))
sns.distplot(df["Item_Weight"])
plt.show()


# In[24]:


plt.figure(figsize=(6,6))
sns.distplot(df["Item_Visibility"])
plt.show()


# In[25]:


plt.figure(figsize=(6,6))
sns.distplot(df["Item_MRP"])
plt.show()


# In[26]:


plt.figure(figsize=(6,6))
sns.distplot(df["Item_Outlet_Sales"])
plt.show()


# In[27]:


plt.figure(figsize=(6,6))
sns.countplot(x="Outlet_Establishment_Year",data=df)
plt.show()


# In[28]:


plt.figure(figsize=(6,6))
sns.countplot(x="Item_Fat_Content",data=df)
plt.show()


# In[29]:


plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type',data=df)
plt.show()


# # Data   Pre-processing

# In[30]:


df.head()


# In[31]:


df["Item_Fat_Content"].value_counts()


# In[32]:


df.replace({'Item_Fat_Content':{'low fat':'Low Fat','LF':'Low Fat','reg':'Regular'}},inplace=True)


# In[33]:


df['Item_Fat_Content'].value_counts()


# In[ ]:




