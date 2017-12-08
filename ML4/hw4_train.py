# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 19:32:48 2017

@author: Ouch
"""

import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer,text_to_word_sequence
import sys

print('Loading Data...')
def load_test_data(file):
    with open(file, encoding="utf-8") as f:
        f.readline()
        return [s[s.index(',')+1:] for s in f.readlines()]

def load_train_data(label_file, no_label_file):
    labels, label_text, unlabel_text = [], [], []
    with open(label_file, encoding="utf-8") as f:
        for s in f.readlines():
            label, text = s.split('+++$+++')
            labels.append(int(label))
            label_text.append(text)
    
    with open(no_label_file,encoding="utf-8") as f:
        unlabel_text = f.readlines()
    return labels, label_text, unlabel_text

label_file = sys.argv[1]
no_label_file = sys.argv[2]

labels, label_text, unlabel_text = load_train_data(label_file, no_label_file)

#把句子拿來做Word2vec
text_train_seq = []
for i in range(len(label_text)):
    text_train_seq.append(label_text[i].split())

##存句子給word2vec
#SentenceFile = open(path+'all_sentences','w', encoding = 'utf8') 
#for row in range(len(all_sentences)):
#    SentenceFile.write(all_sentences[row])
#SentenceFile.close()
#
##讀讀看
#sentenceTable = []
#with open(path+'all_sentences', 'r', encoding = 'utf8') as wFile:
#    for i in wFile.readlines() :
#        i = i.strip('\n')
#        sentenceTable.append([i])

print('Processing--Word2vec--')
from gensim.models import word2vec
import logging
word_dim = 300
max_len_seq = 40

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#word_model = word2vec.Word2Vec([s.split() for s in all_sentences], size=word_dim,min_count=1,sg=1, iter=9)
#word_model.save(path+'models/word2vec2')
#print('word2vec DONE')

#Load Word2vec
print('Loading word2vec...')
word_model = word2vec.Word2Vec.load('word2vec2')
word2index = {word: ind + 1 for ind, word in enumerate(word_model.wv.index2word)} # 0 for padding
word_vectors = [np.zeros(word_dim)]
for word in word_model.wv.index2word:
    word_vectors.append(word_model[word])
word_vectors = np.stack(word_vectors)
text_train = [[word2index.get(s, 0) for s in line] for line in text_train_seq]

print('Training...')
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
text_train = pad_sequences(text_train, maxlen=max_len_seq)

def get_model(word_vectors):
    model = Sequential()
    model.add(Embedding(word_vectors.shape[0], word_vectors.shape[1], trainable=False,
                            embeddings_initializer=keras.initializers.Constant(word_vectors))) # Using pretained word embedding
    model.add(LSTM(500, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


#存最好ㄉ
model = get_model(word_vectors)
checkpoint = ModelCheckpoint('model.hdf5', monitor='val_acc', verbose=0,save_best_only=True, save_weights_only=False, mode='max', period=1)
train_history = model.fit(text_train ,labels ,batch_size = 512, epochs = 20,shuffle='True',validation_split=0.1,callbacks=[checkpoint] )
