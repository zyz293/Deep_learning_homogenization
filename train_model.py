import numpy as np
import pickle
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Activation, Dense
from keras.layers import Input, Flatten, Convolution3D, AveragePooling3D
from keras.callbacks import EarlyStopping
from keras.callbacks import Callback
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.regularizers import l2

print 'load data'
with open('./data.pkl', 'r') as f:
    final_train_data = np.array(pickle.load(f))
with open('./label.pkl', 'r') as f:
    final_train_label = np.array(pickle.load(f))
final_train_data = final_train_data - 0.5
print 'training data shape: ', final_train_data.shape
print 'training label shape: ', final_train_label.shape

print 'create model'
model = Sequential()

model.add(Convolution3D(16, 3, 3, 3, init='glorot_normal', border_mode='same', dim_ordering='tf', W_regularizer=l2(0.001), input_shape=(51,51,51,1)))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(32, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(64, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(128, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2)))
model.add(Convolution3D(256, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001)))
model.add(Activation('relu'))
model.add(AveragePooling3D(pool_size=(2, 2, 2)))

model.add(Flatten())
model.add(Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(0.001)))
model.add(Dense(1, init='glorot_normal', W_regularizer=l2(0.001)))

print 'compile model'
print model.summary()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

print '-------------------------'
print 'fit model'
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(final_train_data, final_train_label, batch_size=8, nb_epoch=2000, validation_split=0.2, callbacks=[early_stopping])

print 'save model'
model.save('my_model.h5')


