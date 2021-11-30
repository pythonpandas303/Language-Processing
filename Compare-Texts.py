# -*- coding: utf-8 -*-
import nltk
from nltk import word_tokenize, sent_tokenize

file_docs = []

with open('Book1.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file_docs.append(line)

print("Number of documents:",len(file_docs))

gen_docs = [[w.lower() for w in word_tokenize(text)] 
            for text in file_docs]
import gensim
dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)

corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]

import numpy as np

tf_idf = gensim.models.TfidfModel(corpus)
for doc in tf_idf[corpus]:
    print([[dictionary[id], np.around(freq, decimals=2)] for id, freq in doc])
    
sims = gensim.similarities.Similarity('C:/Users/your_file_path',tf_idf[corpus],
                                        num_features=len(dictionary))

file2_docs = []

with open ('Book2.txt') as f:
    tokens = sent_tokenize(f.read())
    for line in tokens:
        file2_docs.append(line)

print("Number of documents:",len(file2_docs))  
for line in file2_docs:
    query_doc = [w.lower() for w in word_tokenize(line)]
    query_doc_bow = dictionary.doc2bow(query_doc)
    
query_doc_tf_idf = tf_idf[query_doc_bow]
print('Comparing Result:', sims[query_doc_tf_idf])

sum_of_sims =(np.sum(sims[query_doc_tf_idf], dtype=np.float32))
print(sum_of_sims)

percentage_of_similarity = round(float((sum_of_sims / len(file_docs)) * 100))
print(f'Average similarity float: {float(sum_of_sims / len(file_docs))}')
print(f'Average similarity percentage: {float(sum_of_sims / len(file_docs)) * 100}')
print(f'Average similarity rounded percentage: {percentage_of_similarity}')
