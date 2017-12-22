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

#train = pd.read_csv(train_file).as_matrix()
user = pd.read_csv(user_file).as_matrix()
test = pd.read_csv(test_file).as_matrix()

##shuffle
#data_for_shuffle = train
#np.random.shuffle(data_for_shuffle)
#
##UserID、MovieID、Rating
#y_train = data_for_shuffle[:,3]
#train = data_for_shuffle[:,1:3]

#UserID、MovieID
test = test[:,1:]
#deal with user information
#UserID, Gender, Age, Occupation, ZipCode
user_ID = []
user_information = []
for i in range(len(user)):
    user_information.append(user[i,0].split("::")[1:4])
    gender = user_information[i][0]
    if gender == 'F':
        user_information[i][0] = 0
    else:
        user_information[i][0] = 1
    user_ID.append([user[i,0].split("::")[0]])

user_ID = np.array(user_ID)

user_dict = {}
for i in range(len(user_information)):
    key = user_ID[i,0]
    user_dict[key] = user_information[i]

#print(user_dict['796'])
#deal with movies information   
movies = []
with open (movie_file,encoding="ISO-8859-1") as f:
    next(f)
    movies = f.readlines()
movie_ID = []
movie_dict = {}
movie_type_dict = {}    
movie_information = []
idx = 0
for i in range(len(movies)):
    movie_ID.append([])
    movie_ID[i] = movies[i].split("::")[0]
    movie_information.append(movies[i].split("::"))
    key_type = movie_information[i][2]
    movie_type_dict[key_type] = idx
    idx = idx + 1
    key = movie_information[i][0]
    del movie_information[i][0]
    movie_dict[key] = movie_information[i]


##print(len(train[:,0]))   
##check data with user information
#train_inDetail = []
#movie_id_type = []
#for i in range(len(train[:,0])):
#    #Check UserID with user_dict
#    userID = train[i,0]
#    train_inDetail.append([])
#    train_inDetail[i] = user_dict[str(userID)]
#    #Check MovieID with movie_dict&type_dict
#    movieID = train[i,1]
#    movie_id_ = movie_dict[str(movieID)]
#    t = movie_id_[1]
#    movie_type_ = movie_type_dict[str(t)]
#    movie_id_type.append([movie_type_])
#    #movie_id_type[i]= movie_type_
#    train_inDetail[i] = [float(x) for x in train_inDetail[i]]
movie_ID = np.array(movie_ID, dtype= int)

t_movie_id_type = []
test_inDetail = []
#print(len(test[:,0]))
for i in range(len(test[:,0])):
    userID = test[i,0]
    test_inDetail.append([])
    test_inDetail[i] = user_dict[str(userID)] 
    #Check MovieID with movie_dict&type_dict
    movieID = test[i,1]
    movie_id_ = movie_dict[str(movieID)]
    t = movie_id_[1]
    movie_type_ = movie_type_dict[str(t)]
    t_movie_id_type.append([movie_type_])
    #movie_id_type[i]= movie_type_
    test_inDetail[i] = [float(x) for x in test_inDetail[i]]
    
test_inDetail = np.array(test_inDetail,dtype = float)
t_movie_id_type = np.array(t_movie_id_type, dtype = float)   
#train_inDetail = np.array(train_inDetail,dtype = float)
#movie_id_type = np.array(movie_id_type,dtype = float)
#UserID、MovieID、Gender、Age、Occupation、MovieType
#train_in = np.concatenate((train_inDetail, movie_id_type), axis = 1)
#x_train = np.concatenate((train, train_in) ,axis = 1)



#x_train_movieID = x_train[:,1]
#x_train_user = np.delete(x_train,1,1)
test_in = np.concatenate((test,test_inDetail ) ,axis = 1)
x_test = np.concatenate((test_in ,t_movie_id_type),axis = 1)
x_test_movieID = x_test[:,1]
x_test_user = np.delete(x_test,1,1)

print('---Done with data---')

from keras.models import Sequential, Model, load_model
from keras.layers import add, Input, Dense, Dropout, Flatten, Activation, Reshape, Concatenate
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

#print('---Training DNN Model---')
#def DNN_model(users,movies,latent_dim ):
#    user_input = Input(shape = [5])
#    item_input = Input(shape = [1])
#    user_vec = Embedding(users, latent_dim, embeddings_initializer = 'random_normal')(user_input)
#    user_vec = Flatten()(user_vec)
#    item_vec = Embedding(movies, latent_dim, embeddings_initializer = 'random_normal')(item_input)
#    item_vec = Flatten()(item_vec)
#    merge_vec = Concatenate()([user_vec, item_vec])     
#    hidden = Dense(512,activation = 'relu')(merge_vec)     
#    drop = Dropout(0.5)(hidden)     
#    hidden = Dense(256,activation = 'linear' )(drop)     
#    drop = Dropout(0.4)(hidden)     
#    hidden = Dense(256,activation = 'linear' )(drop)     
#    drop = Dropout(0.3)(hidden)     
#    hidden = Dense(256,activation = 'linear' )(drop)     
#    output = Dense(1)(hidden)  
#    model = Model([user_input, item_input], output)
#    model.compile(loss = 'mse', optimizer = 'adam' ,metrics = ['mse'])
#    model.summary()
#    return model
#
#latent_dim =444 #444
#print(len(user_ID))
#u_input = len(user_ID)+ 1
#m_input = len(movie_ID) + 1
#model = DNN_model(u_input, m_input, latent_dim)

#x_train_user = x_train_user.tolist()
#x_train_movieID = x_train_movieID.tolist()

#checkpoint = ModelCheckpoint(path+'models/model_loss{val_loss:.4f}_epoch{epoch:03d}.hdf5', monitor='val_loss', verbose=0,save_best_only=False, save_weights_only=False, mode='min', period=1)
#train_history = model.fit( [x_train_user,x_train_movieID] ,y_train ,batch_size = 1024, epochs =20,shuffle='True',validation_split=0.1,callbacks=[checkpoint] )

#x_test_user = x_test_user.tolist()
#x_test_movieID = x_test_movieID.tolist()

model =load_model('model_loss0.7312.hdf5')

#model.load_weights(path + 'models/model_loss0.7312_epoch010.hdf5')

y_test = model.predict( [x_test_user,x_test_movieID] )
y_test = np.clip(y_test, 1,5)

print(y_test)
print('---Writing CSV---')

y_test = np.array(y_test)
print(len(y_test))
#df = pd.DataFrame({ 'Rating': y_test,'TestDataID': np.arange(x_test.shape[0])+1}).to_csv('Predict_DNN.csv', index=False)

#outputFile = 'Predict_DNN.csv'

with open(outputFile, 'w') as file:
    file.write('TestDataID,Rating\n')
    for i, y in  enumerate(y_test):
        file.write('%d,%.20f\n' %(i+1, y))