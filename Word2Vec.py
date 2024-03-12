#!/usr/bin/env python
# coding: utf-8

# In[1]:


import gensim
import pandas as pd


# In[3]:


df = pd.read_json("E:\\reviews_Cell_Phones_and_Accessories_5.json\\Cell_Phones_and_Accessories_5.json", lines=True)
df


# In[4]:


df.shape


# In[7]:


df.reviewText[0]


# In[10]:


review_text=df.reviewText.apply(gensim.utils.simple_preprocess)
review_text


# In[12]:


model=gensim.models.Word2Vec(
window=10,
min_count=2,
workers=4)


# In[14]:


model.build_vocab(review_text,progress_per=1000)


# In[15]:


model.epochs


# In[16]:


model.corpus_count


# In[17]:


model.train(review_text,total_examples=model.corpus_count,epochs=model.epochs)


# In[ ]:




