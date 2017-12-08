# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:05:02 2017

@author: Ouch
"""
import time
start = time.time()
from keras.models import model_from_json, load_model
import pandas as pd
import numpy as np
import sys
from keras.preprocessing.text import Tokenizer,text_to_word_sequence

test_file = sys.argv[1]
outputFile = str(sys.argv[2])

print('Loading Test Data...')
def load_test_data(file):
    with open(file, encoding="utf-8") as f:
        f.readline()
        return [s[s.index(',')+1:] for s in f.readlines()]

test_text = load_test_data(test_file)

text_test_seq = []
for i in range(len(test_text)):
    text_test_seq.append(test_text[i].split())
print('Loading word2vec...')
from gensim.models import word2vec
import logging
word_dim = 300
max_len_seq = 40

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

word_model = word2vec.Word2Vec.load('word2vec2')
word2index = {word: ind + 1 for ind, word in enumerate(word_model.wv.index2word)} # 0 for padding
word_vectors = [np.zeros(word_dim)]
for word in word_model.wv.index2word:
    word_vectors.append(word_model[word])
word_vectors = np.stack(word_vectors)
text_test = [[word2index.get(s, 0) for s in line] for line in text_test_seq]

import keras
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM, Dropout
from keras.datasets import imdb
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

max_len_seq = 40
text_test = pad_sequences(text_test, maxlen=max_len_seq)

def get_model(word_vectors):
    model = Sequential()
    model.add(Embedding(word_vectors.shape[0], word_vectors.shape[1], trainable=False,
                            embeddings_initializer=keras.initializers.Constant(word_vectors))) # Using pretained word embedding
    model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model
model = get_model(word_vectors)
model.load_weights('model.hdf5')
print('Predicting...')
y_test = model.predict_classes(text_test, batch_size=256, verbose=1)
#y_test = model.predict(text_test, batch_size=256, verbose=1)

print('Writing...')
df = pd.DataFrame({'label':y_test.T[0], 'id': range(len(y_test))}).to_csv(outputFile, index = False)

end = time.time()
t = end - start
print('Testing Time: ' + str(t) )