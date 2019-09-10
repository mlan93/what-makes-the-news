#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import spacy
import pickle
import en_core_web_sm
from spacy_readability import Readability

with open('local_pickle/article_df.pkl', 'rb') as f:
    original_article_df = pickle.load(f)

nlp = en_core_web_sm.load()
nlp.add_pipe(Readability())

metric_list = []

count = 0

for art in original_article_df['content']:
    print(count)
    doc = nlp(art)
    metric_list.append([doc._.flesch_kincaid_grade_level, doc._.flesch_kincaid_reading_ease,                        doc._.dale_chall, doc._.smog, doc._.coleman_liau_index,                        doc._.automated_readability_index, doc._.forcast])
    count += 1

print(metric_list)
    
# with open('metric_list_AWS.pkl', 'wb') as f:
#     pickle.dump(metric_list, f)

