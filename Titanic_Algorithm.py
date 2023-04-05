#!/usr/bin/env python
# coding: utf-8

# In[16]:


import pandas as pd
import numpy as np


# In[17]:


titanic = pd.read_csv ('desktop/Trainset.csv')
titanic.head()


# In[18]:


titanic = titanic [['Survived','Pclass','Sex','Age']]
titanic.dropna (axis=0, inplace=True)
titanic ['Sex'].replace(['male','female'],[0,1], inplace=True)
titanic.head()


# In[19]:


from sklearn.neighbors import KNeighborsClassifier


# In[20]:


X = titanic.drop ('Survived',axis=1)
y = titanic ['Survived']


# In[21]:


score = []
best_k = 1
best_score = 0

for k in range (best_k,30):
    model = KNeighborsClassifier(k)
    model.fit(X,y)
    score.append(model.score(X,y))
    
    if best_score < (model.score(X,y)):
        best_k = k
        best_score = model.score(X,y)
        
print(best_k)


# In[22]:


model.fit (X,y)
model.score (X,y)


# In[23]:


def Survie(model, Pclass=1, Sex=1, Age=25):
    x = np.array([Pclass, Sex, Age]).reshape(1,3)
    print(model.predict(x))
    print(model.predict_proba(x))


# In[24]:


Survie(model)


# In[ ]:




