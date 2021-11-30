## Importing Required Libraries ##
import re
import string
import numpy as np
import random
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
%matplotlib inline
from plotly import graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
from collections import Counter

from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from tqdm import tqdm
import os
import nltk
import spacy
import random
from spacy.util import compounding
from spacy.util import minibatch

from collections import defaultdict
from collections import Counter

import keras
from keras.models import Sequential
from keras.initializers import Constant
from keras.layers import (LSTM, 
                          Embedding, 
                          BatchNormalization,
                          Dense, 
                          TimeDistributed, 
                          Dropout, 
                          Bidirectional,
                          Flatten, 
                          GlobalMaxPool1D)
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau


from sklearn.metrics import (
    precision_score, 
    recall_score, 
    f1_score, 
    classification_report,
    accuracy_score,
    confusion_matrix
)
from sklearn.preprocessing import LabelEncoder

## Setting color palette ##
primary_blue = "#00ffff"
primary_blue2 = "#0000ff"
primary_blue3 = "#000080"
primary_grey = "#778899"
primary_black = "#253529"
primary_bgcolor = "#2b2b2b"

primary_green = px.colors.qualitative.Plotly[2]

## Importing data, in repo as 'spam.csv' ##
spam = pd.read_csv('spam.csv', encoding='latin-1')

## Dropping NA's and renaming columns ##
spam = spam.dropna(how="any", axis=1)
spam.columns = ['class', 'message']

## Adding column to indicate message length ##
spam['message_length'] = spam['message'].apply(lambda x: len(x.split(' ')))

## Producing a count of classes ##
balance_counts = spam.groupby('class')['class'].agg('count').values

## Show count of classes ##
balance_counts

## Plot of classes ##
fig = go.Figure()
fig.add_trace(go.Bar(
    x=['ham'],
    y=[balance_counts[0]],
    name='ham',
    text=[balance_counts[0]],
    textposition='auto',
    marker_color=primary_blue
))
fig.add_trace(go.Bar(
    x=['spam'],
    y=[balance_counts[1]],
    name='spam',
    text=[balance_counts[1]],
    textposition='auto',
    marker_color=primary_grey
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Dataset distribution by Class</span>'
)
fig.show()

## Plot message length by class ##
ham_ind = spam[spam['class'] == 'ham']['message_length'].value_counts().sort_index()
spam_ind = spam[spam['class'] == 'spam']['message_length'].value_counts().sort_index()

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=ham_ind.index,
    y=ham_ind.values,
    name='ham',
    fill='tozeroy',
    marker_color=primary_blue,
))
fig.add_trace(go.Scatter(
    x=spam_ind.index,
    y=spam_ind.values,
    name='spam',
    fill='tozeroy',
    marker_color=primary_grey,
))
fig.update_layout(
    title='<span style="font-size:32px; font-family:Times New Roman">Message Length by Class</span>'
)
fig.update_xaxes(range=[0, 70])
fig.show()

## Function to remove punctuation and convert all letters to lowercase ##
def clean_text(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

## Adding a new column to house clean text ##
spam['message_clean'] = spam['message'].apply(clean_text)

## Adding stopword from dict, as well as common text usage ##
stop_words = stopwords.words('english')
more_stopwords = ['u', 'im', 'c']
stop_words = stop_words + more_stopwords

## Function to remove stopwords ##
def remove_stopwords(text):
    text = ' '.join(word for word in text.split(' ') if word not in stop_words)
    return text

## Function to apply stemming ##
stemmer = nltk.SnowballStemmer("english")

def stemm_text(text):
    text = ' '.join(stemmer.stem(word) for word in text.split(' '))
    return text

## Applying stemming ##
spam['message_clean'] = spam['message_clean'].apply(stemm_text)

## Encoding Classes ##
le = LabelEncoder()
le.fit(spam['class'])
spam['class_encoded'] = le.transform(spam['class'])

## Wordcloud for Ham messages ##
wc = WordCloud(
    background_color='white', 
    max_words=200
)
wc.generate(' '.join(text for text in spam.loc[spam['class'] == 'ham', 'message_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for HAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()

## Wordcloud for Spam messages ##
wc = WordCloud(
    background_color='white', 
    max_words=200
)
wc.generate(' '.join(text for text in spam.loc[spam['class'] == 'spam', 'message_clean']))
plt.figure(figsize=(18,10))
plt.title('Top words for SPAM messages', 
          fontdict={'size': 22,  'verticalalignment': 'bottom'})
plt.imshow(wc)
plt.axis("off")
plt.show()

## Setting variables for training ##
x = spam['message_clean']
y = spam['class_encoded']

## Train/Test split ##
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
print(len(x_train), len(y_train))
print(len(x_test), len(y_test))

## Vectorizing x_train data ##
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(x_train)

## Vectorizing ##
x_train_dtm = vect.transform(x_train)
x_test_dtm = vect.transform(x_test)

## Tuning vector ##
vect_tunned = CountVectorizer(stop_words='english', ngram_range=(1,2), 
min_df=0.1, max_df=0.7, max_features=100)

## Initiating the TF-IDF ##
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
tfidf_transformer.fit(x_train_dtm)
x_train_tfidf = tfidf_transformer.transform(x_train_dtm)

## Results of TF-IDF ##
x_train_tfidf

## Importing Multinomial Bayesian Model ##
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()

## Fitting model to training data ##
nb.fit(x_train_dtm, y_train)

## Applying model to data ##
y_pred_class = nb.predict(x_test_dtm)
y_pred_prob = nb.predict_proba(x_test_dtm)[:, 1]

## Confusion matrix of predictions ##
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))
confusion_matrix(y_test, y_pred_class)

## XGBoost on same data set ##
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

pipe = Pipeline([('bow', CountVectorizer()), 
                 ('tfid', TfidfTransformer()),  
                 ('model', XGBoost())])

## Fitting model and testing accuracy ##
pipe.fit(x_train, y_train)
y_pred_class = pipe.predict(x_test)
print(metrics.accuracy_score(y_test, y_pred_class))

## Confusion matrix on XGBoost ##
confusion_matrix(y_test, y_pred_class)

## Displaying top 5 words in HAM messages ##
textnum = spam.loc[spam['class'] == 'ham', 'message_clean'].str.split(expand=True).stack().value_counts()
print(textnum)

## Displaying top 5 words in SPAM messages ##
textnum2 = spam.loc[spam['class'] == 'spam', 'message_clean'].str.split(expand=True).stack().value_counts()
print(textnum2)

## Plotting accuracy of both models ##
import matplotlib.pyplot as plt
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]//2, y[i], ha = 'center')
if __name__ == '__main__':
    
    x = ["Multinomial Naive Bayes", "XGBoost"]
    y = [97.85, 95.98]
    plt.figure(figsize = (10, 5))
    plt.bar(x, y)
    addlabels(x, y)
    plt.title("Model Accuracy")
    plt.xlabel("Model Type")
    plt.ylabel("Accuracy")
    plt.show()

