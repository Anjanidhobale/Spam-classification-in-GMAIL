#!/usr/bin/env python
# coding: utf-8

# In[122]:


import pandas as pd
import numpy as np
import en_core_web_sm


# In[97]:


nlp = en_core_web_sm.load()


# In[121]:


data = pd.read_csv('https://techlearn-cdn.s3.amazonaws.com/project_GmailSpamClassification/spam.csv',encoding='cp1252')


# In[99]:


data.head()


# In[100]:


data = data[['v1','v2']]


# In[101]:


data['v1'] = data['v1'].apply(lambda x:0 if x=='ham' else 1)


# In[102]:


data


# In[103]:


### Text Pre-Processing
def process(x):
    temp = []
    document = nlp(x.lower())
    print(document)
    for i in document:
        if i.is_stop!=True and i.is_punct!= True:
            print(i)
            temp.append(i.lemma_)
            print(temp)
        else:
            pass
        
    return (' '.join(temp))
##


# In[104]:


data['v2'] = data['v2'].apply(lambda x: process(x))


# In[105]:


data.head()


# In[106]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[107]:


vectorizer = TfidfVectorizer(analyzer='word',stop_words='english')


# In[108]:


text_vector = vectorizer.fit_transform(data['v2'].values.tolist())


# In[109]:


print(text_vector)


# In[110]:


######Splitting Data set


# In[111]:


from sklearn.model_selection import train_test_split


# In[112]:


x_train, x_test, y_train, y_test = train_test_split(text_vector.toarray(),data['v1'],test_size=0.2,random_state=20)


# In[113]:


len(x_test)


# In[89]:


####Model Building


# In[114]:


from sklearn.naive_bayes import BernoulliNB


# In[115]:


modelB = BernoulliNB()
modelB.fit(x_train,y_train)
print(modelB.score(x_train,y_train))


# In[116]:


y_predictedB = modelB.predict(x_test)


# In[117]:


from sklearn.metrics import accuracy_score


# In[94]:


print(accuracy_score(y_test,y_predictedB))


# In[ ]:




