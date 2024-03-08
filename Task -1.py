#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
import warnings
warnings.simplefilter("ignore")


# In[2]:


sns.set()


# # Load data

# In[3]:


data= datasets.load_iris()


# In[4]:


data.keys()


# In[5]:


print(data[('DESCR')])


# In[6]:


data


# In[7]:


data["data"] [: 5]


# In[8]:


data["feature_names"]


# In[9]:


data["target"]


# In[10]:


data["target_names"]


# In[11]:


df=pd.DataFrame(data['data'] , columns=data['feature_names'])


# In[12]:


df['target']=data['target']


# In[13]:


df.head()


# # Basic descriptive statistics
# 

# In[14]:


df.describe()


# # Distribution of features and targets

# In[15]:


col ="sepal length (cm)"
df[col].hist()
plt.suptitle(col)
plt.show()


# In[16]:


col ="sepal width (cm)"
df[col].hist()
plt.suptitle(col)
plt.show()


# In[17]:


col ="petal width (cm)"
df[col].hist()
plt.suptitle(col)
plt.show()


# # Relationship of the data features with the target

# In[18]:


df["target_name"]=df["target"].map({0:"setosa",1:"versicolor",2:"virginica"})


# In[19]:


df.head()


# In[20]:


col ="sepal length (cm)"
sns.relplot ( x=col,y ='target',hue="target_name",data=df)
plt.suptitle(col,y=1.05)
plt.show()


# In[21]:


col ="sepal width (cm)"
sns.relplot ( x=col,y ='target',hue="target_name",data=df)
plt.suptitle(col,y=1.05)
plt.show()


# In[22]:


col ="petal length (cm)"
sns.relplot ( x=col,y ='target',hue="target_name",data=df)
plt.suptitle(col,y=1.05)
plt.show()


# In[23]:


col ="petal width (cm)"
sns.relplot ( x=col,y ='target',hue="target_name",data=df)
plt.suptitle(col,y=1.05)
plt.show()


# # Exploratory Data Analysis -pairplots

# In[24]:


sns.pairplot(df)


# In[25]:


sns.pairplot(df,hue='target_name')


# In[26]:


df_encoded = pd.get_dummies(df)
corr = df_encoded.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values)
plt.show()


# # Modeling Logistic Regression

# In[27]:


from sklearn.linear_model import LogisticRegression


# In[28]:


model=LogisticRegression() 


# In[29]:


X=df.iloc[:,:4]
y=df.iloc[:, 4 ]


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[31]:


model.fit(X_train, y_train)


# In[32]:


y_pred=model.predict(X)


# In[33]:


y_pred


# In[34]:


model.score(X_train, y_train)


# In[ ]:




