# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:11:46 2017

@author: Ouch
"""
#Load model and word2vec
wget -O model.hdf5 https://www.dropbox.com/s/5utf17bi016s83i/model.hdf5?dl=1
wget -O word2vec2.syn1neg.npy https://www.dropbox.com/s/8muj76too4eo4vk/word2vec2.syn1neg.npy?dl=1
wget -O word2vec2.wv.syn0.npy https://www.dropbox.com/s/o4z8jt87gs2rur0/word2vec2.wv.syn0.npy?dl=1

python hw4_test.py $1 $2