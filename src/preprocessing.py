import numpy as np
import pandas as pd
import string
import re
import nltk
import pickle

def remove_punctuation(row):
    return ''.join([char for char in row if char not in string.punctuation])

def tokenize(row):
    return re.split('\W+',row)

def remove_stopwords(row):
    return [word for word in row if word not in stopwords]

def stemming(porterStemmer, row):
    return [porterStemmer.stem(word) for word in row]

def createInputs(vocab_size, text):    
    inputs = []
    for w in text:
        v = np.zeros((vocab_size, 1))
        v[word_to_idx[w]] = 1
        inputs.append(v)
    
    return inputs

stopwords = nltk.corpus.stopwords.words('english')
porterStemmer = nltk.PorterStemmer()

df = pd.read_csv('../data/spam.csv',delimiter=',',encoding='latin-1')
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'],axis=1,inplace=True)
df.columns=['label','text']
ham_data = (df[df.label == 'ham']).head(50)
spam_data = (df[df.label == 'spam']).head(50)

data = pd.concat([ham_data, spam_data],axis=0)
data = data.reset_index(drop=True)
del df

data['encoded_text']=data['text'].apply(lambda row : remove_punctuation(row))
data['encoded_text']=data['encoded_text'].apply(lambda row : tokenize(row.lower()))
data['encoded_text']=data['encoded_text'].apply(lambda row : remove_stopwords(row))
data['encoded_text']=data['encoded_text'].apply(lambda row : stemming(porterStemmer, row))

words = []
for row in data['encoded_text']:
    for word in row:
        if word not in words:
            words.append(word)

vocab_size = len(words)

word_to_idx = { w: i for i, w in enumerate(words) }
idx_to_word = { i: w for i, w in enumerate(words) }

data['encoded_text'] = data['encoded_text'].apply(lambda row : createInputs(vocab_size, row))
encoded_label = pd.get_dummies(data['label'])
data = pd.concat([data, encoded_label],axis=1)

data.to_pickle('../data/encoded_spam.pkl')

