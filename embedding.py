import numpy as np
np.random.seed(42)
import pandas as pd
import re
import pickle

from keras.preprocessing import text, sequence

import warnings
warnings.filterwarnings('ignore')

import os
os.environ['OMP_NUM_THREADS'] = '4'

def normalize(s):
    s = s.lower()
    # Replace ips
    s = re.sub(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}', ' _ip_ ', s)
    # Isolate punctuation
    s = re.sub(r'([\'\"\.\(\)\!\?\-\\\/\,])', r' \1 ', s)
    # Remove some special characters
    s = re.sub(r'([\;\:\|•«\n])', ' ', s)
    # Replace numbers and symbols with language
    s = s.replace('&', ' and ')
    s = s.replace('@', ' at ')
    s = s.replace('0', ' zero ')
    s = s.replace('1', ' one ')
    s = s.replace('2', ' two ')
    s = s.replace('3', ' three ')
    s = s.replace('4', ' four ')
    s = s.replace('5', ' five ')
    s = s.replace('6', ' six ')
    s = s.replace('7', ' seven ')
    s = s.replace('8', ' eight ')
    s = s.replace('9', ' nine ')
    
    # some cleaning 
    s = re.sub(r"what's", "what is ", s)
    s = re.sub(r"\'s", " ", s)
    s = re.sub(r"\'ve", " have ", s)
    s = re.sub(r"can't", "cannot ", s)
    s = re.sub(r"n't", " not ", s)
    s = re.sub(r"i'm", "i am ", s)
    s = re.sub(r"\'re", " are ", s)
    s = re.sub(r"\'d", " would ", s)
    s = re.sub(r"\'ll", " will ", s)
    s = re.sub(r"\'scuse", " excuse ", s)
    s = re.sub('\W', ' ', s)
    s = re.sub('\s+', ' ', s)
    # remove urls
    s = re.sub(r'^https?:\/\/.*[\r\n]*', '', s)
    s = re.sub(r"www\S+", "", s)
    s = s.strip(' ')
    return s

def normalize_array(a):
    for x, value in np.ndenumerate(a):
        a[x] = normalize(value)
    return a




'''
Main Program:
Generate embedding matrix for fastText, glove, glove(twitter)
'''

# some global variables
max_features = 100000
maxlen = 200
embed_size = 300

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

X_train = train["comment_text"].fillna("_NA_").values
y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
y_train_toxic = train["toxic"].values
y_train_severe_toxic = train["severe_toxic"].values
y_train_obscene = train["obscene"].values
y_train_threat = train["threat"].values
y_train_identity_hate = train["identity_hate"].values
X_test = test["comment_text"].fillna("_NA_").values

X_train = normalize_array(X_train)
X_test = normalize_array(X_test)

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(X_train) + list(X_test))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)
pickle.dump(x_train, open("x_train.pickle", "wb"))
pickle.dump(y_train, open("y_train.pickle", "wb"))
pickle.dump(y_train_toxic, open("y_train_toxic.pickle", "wb"))
pickle.dump(y_train_severe_toxic, open("y_train_severe_toxic.pickle", "wb"))
pickle.dump(y_train_obscene, open("y_train_obscene.pickle", "wb"))
pickle.dump(y_train_threat, open("y_train_threat.pickle", "wb"))
pickle.dump(y_train_identity_hate, open("y_train_identity_hate.pickle", "wb"))
pickle.dump(x_test, open("x_test.pickle", "wb"))


def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')

def generate_embedding_matrix(embedding_file, out_pickle):
    embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(embedding_file))
    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: 
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # save to pickle
    pickle.dump(embedding_matrix, open(out_pickle, "wb"))

# fastText
generate_embedding_matrix("fastText.300d.vec", "fastText.300d.pickle")
# Glove
generate_embedding_matrix("glove.840B.300d", "glove.300d.pickle")
# Glove(twitter)
generate_embedding_matrix("glove.twitter.27B.200d.txt", "glove.twitter.200d.pickle")