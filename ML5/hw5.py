# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:03:38 2017

@author: Ouch
"""

#path = 'C:/Users/HongLab/Desktop/kiki/5/'
import pandas as pd
import numpy as np
import sys
test_file = sys.argv[1]
outputFile = sys.argv[2]
movie_file = sys.argv[3]
user_file = sys.argv[4]

#train_file = sys.argv[1]
#train = pd.read_csv(train_file).as_matrix()
user = pd.read_csv(user_file).as_matrix()
test = pd.read_csv(test_file).as_matrix()
x_test =test[:,1]
x_test_movieID = test[:,2]

##shuffle
#data_for_shuffle = train
#np.random.shuffle(data_for_shuffle)

##UserID、MovieID、Rating
#y_train = data_for_shuffle[:,3]
#x_train = data_for_shuffle[:,1]
#x_train_movieID = data_for_shuffle[:,2]

##Normalized
#std_ = np.std(y_train,axis = 0)
#mean_ = np.mean(y_train,axis = 0)
#for i in range(len(y_train)):
#    y_train[i] = ( y_train[i] - mean_ ) / std_

print('---Done with data---')

from keras.models import Sequential, Model, load_model
from keras.layers import Add, Input, Dense, Dropout, Flatten, Activation, Reshape, Concatenate
from keras.layers.wrappers import Bidirectional

from keras.optimizers import SGD, Adam

from keras import regularizers
from keras.regularizers import l2
from keras.initializers import Zeros
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.layers import BatchNormalization, Merge, Dot, TimeDistributed, Lambda
from keras.preprocessing import sequence
from keras.preprocessing.sequence import pad_sequences
from keras.layers.embeddings import Embedding

#print('---Training MF Model---')
#def MF_model(users,movies,latent_dim ):
#    user_input = Input(shape = [1])
#    item_input = Input(shape = [1])
#    user_vec = Embedding(users, latent_dim, embeddings_initializer = 'random_normal')(user_input)
#    user_vec = Flatten()(user_vec)
#    item_vec = Embedding(movies, latent_dim, embeddings_initializer = 'random_normal')(item_input)
#    item_vec = Flatten()(item_vec)
#    
#    user_bias = Embedding(users,1, embeddings_initializer = 'zeros')(user_input)
#    user_bias = Flatten()(user_bias)
#    item_bias = Embedding(movies,1, embeddings_initializer = 'zero')(item_input)
#    item_bias = Flatten()(item_bias)
#    
#    r_hat = Dot(axes = 1)( [user_vec, item_vec] )
#    r_hat = Add()([r_hat, user_bias, item_bias])
#    model = Model([user_input, item_input], r_hat)
#    model.compile(loss = 'mse', optimizer = 'adagrad' ,metrics = ['mse'])
#    model.summary()
#    return model
#
#
#latent_dim =222
##print(len(user_ID))
#u_input = np.max(x_train)+ 1
#m_input = np.max(x_train_movieID) + 1
#model = MF_model(u_input, m_input, latent_dim)


#x_train_user = x_train_user.tolist()
#x_train_movieID = x_train_movieID.tolist()

#checkpoint = ModelCheckpoint(path+'models/MFmodel_loss{val_loss:.4f}_epoch{epoch:03d}.hdf5', monitor='val_loss', verbose=0,save_best_only=True, save_weights_only=False, mode='min', period=3)
#train_history = model.fit( [x_train,x_train_movieID] ,y_train ,batch_size = 1024, epochs =30,shuffle='True',validation_split=0.1,callbacks=[checkpoint] )
#
##x_test_user = x_test_user.tolist()
##x_test_movieID = x_test_movieID.tolist()
#

model =load_model('MFmodel_loss0.7379.hdf5')
#
y_test = model.predict( [x_test,x_test_movieID] )
##for i in range(len(y_test)):
##    y_test[i] = y_test[i]*std_ + mean_
y_test = np.clip(y_test, 1,5)
#print(y_test)
print('---Writing CSV---')

#y_test = np.array(y_test)
##print(len(y_test))
###df = pd.DataFrame({ 'Rating': y_test,'TestDataID': np.arange(x_test.shape[0])+1}).to_csv('Predict_DNN.csv', index=False)
##
#outputFile = 'Predict_MF_7379.csv'
#
with open(outputFile, 'w') as file:
    file.write('TestDataID,Rating\n')
    for i, y in  enumerate(y_test):
        file.write('%d,%.20f\n' %(i+1, y))