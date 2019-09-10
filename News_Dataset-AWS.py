#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[ ]:


import pandas as pd
import pickle
import string
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize

nltk.download('punkt')
nltk.download('wordnet')

with open('article_df_AWS.pkl', 'rb') as f:
    article_df = pickle.load(f)
    
# Remove punctuation:
# count = 0
# for index, row in article_df.iterrows():
#     count += 1
#     print(count)
#     article_df.loc[index, 'content'] = (article_df.loc[index, 'content']
#                                         .translate(str.maketrans('', '', string.punctuation)))

# with open('article_df_AWS.pkl', 'wb') as f:
#     pickle.dump(article_df, f)
    
# Lemmatize data
lemma = WordNetLemmatizer()

count = 0
for index, row in article_df.iterrows():
    count += 1
    print(count)
    article_df.loc[index, 'content'] = ' '.join([lemma.lemmatize(word) for word in 
                                                 word_tokenize(article_df.loc[index, 'content'])])

with open('article_df_AWS_lemma.pkl', 'wb') as f:
    pickle.dump(article_df, f)

