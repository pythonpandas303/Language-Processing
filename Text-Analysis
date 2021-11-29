# -*- coding: utf-8 -*-
import operator
import time
import string
import re
from collections import defaultdict
from nltk.corpus import stopwords

try:
    stops = stopwords.words('english')
except LookupError:
    import nltk
    nltk.download('stopwords')

stops = set(stops)
filepath = 'C:\\Users\\your_file_path\\'
f = open(filepath+'Book1.txt', 'rb')
start = time.time()
your_book = defaultdict(int)
punc = string.punctuation

for line in f:
    cln_line = re.sub('[' + punc + ']', '', line.decode('utf‐8'))
    cln_line = cln_line.lower()
    spl_line=cln_line.split()
    for word in spl_line:
        
        if word in stops: 
            continue
        lower_word = word.lower()
        your_book.setdefault(lower_word, 0)
        your_book[lower_word] += 1
        your_book.setdefault(word, 0)
        your_book[word] += 1
    
sorted_your_book = sorted(your_book.items(), key=operator.itemgetter(1), reverse=True)
elapsed = time.time() - start

print('Run took', elapsed, ' seconds.')
print('Number of distinct words: ', len(sorted_your_book))

top_n = 20
y = []

for pair in range (top_n):
    y.append([sorted_your_book[pair][1]])
    print(sorted_your_book[pair])
