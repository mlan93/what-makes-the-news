#!/usr/bin/env python
# coding: utf-8

# ### Import packages

# In[ ]:


import pandas as pd
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# gensim
from gensim import corpora, models, similarities, matutils

# Load data
with open('article_df_AWS_lemma.pkl', 'rb') as f:
    article_df = pickle.load(f)

with open('article_countvec_lemma.pkl', 'rb') as f:
    count_vectorizer = pickle.load(f)

with open('doc_word.pkl', 'rb') as f:
    doc_word = pickle.load(f)

# # Convert the text to all lower case.
# article_list = article_df['content'].str.lower()

# # Add additional stop words that commonly occur but don't add much value (including common news terms and key
# # words in the names of the publications)
# add_stop_words = set(['said', 'say', 'says', 'dont', 'post', 'news', 'breitbart', 'buzzfeed', 'cnn', 'guardian', \
#                       'times', 'npr', 'reuters', 'vox', 'according', 'have', 'ha', 'wa', 'mr', 'ms', 'mrs', \
#                       'just', 'w', 'p', ])
# stop_words = ENGLISH_STOP_WORDS.union(add_stop_words)

# # Create a CountVectorizer for parsing/counting words
# # https://regex101.com/r/PJsv3u/1 to show example of the token pattern

# # ngram 1-2, as the original text within the data replaced all apostrophes with spaces already, resulting in
# # conjunctions being ignored by the vectorizer when using ngram 1. Also, there are many important two-word phrases
# # in news, like proper names and other important items.

# # max_df=0.99 to exclude terms that show up in more than 95% of documents
# # min_df=0.01 to exclude terms that show up in less than 1% of documents

# count_vectorizer = CountVectorizer(ngram_range=(1, 2),  
#                                    stop_words=stop_words, token_pattern=r"\b[a-zA-Z]+\b",
#                                    max_df = 0.99, min_df = 0.01)

# count_vectorizer.fit(article_list)

# # Pickle the completed CountVectorizer as article_countvec_lemma.pkl
# with open('article_countvec_lemma.pkl', 'wb') as f:
#     pickle.dump(count_vectorizer, f)

# # Create the term-document matrix
# # Transpose it so the terms are the rows and the articles are the column numbers
# doc_word = count_vectorizer.transform(article_list).transpose()
# doc_word_T = count_vectorizer.transform(article_list)

# # Pickle the completed matrices as doc_word.pkl and doc_word_T.pkl
# with open('doc_word.pkl', 'wb') as f:
#     pickle.dump(doc_word, f)

# with open('doc_word_T.pkl', 'wb') as f:
#     pickle.dump(doc_word_T, f)

# # Convert sparse matrix of counts to a gensim corpus
# corpus = matutils.Sparse2Corpus(doc_word)

# corpora.MmCorpus.serialize('corpus.mm', corpus)

corpus = corpora.MmCorpus('corpus.mm')

id2word = dict((v, k) for k, v in count_vectorizer.vocabulary_.items())

def try_LDA(corpus, id2word, num_topics):
    count = 1
    print(count)
    for n in range(31, num_topics + 1):
        lda = models.LdaMulticore(corpus=corpus, num_topics=n, id2word=id2word, passes=5,                                   random_state = 42)
        with open(f'LDA/LDA_model_{n}.pkl', 'wb') as f:
            pickle.dump(lda, f)
        count +=1
        print(count)

try_LDA(corpus, id2word, 50)

