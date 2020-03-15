# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import EarlyStopping
import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from sklearn.neighbors import NearestCentroid
import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import itertools
from random import sample 

n_steps_in, n_steps_out = 3, 2
# covert into input/output
X = np.load('data/X.data.npy')
y = np.load('data/y.data.npy')
print(X.shape)
# the dataset knows the number of features, e.g. 2
n_features = X.shape[2]

# define model
model = Sequential()
model.add(LSTM(4096, activation='relu', input_shape=(n_steps_in, n_features), dropout=0.4, recurrent_dropout=0.2))
model.add(RepeatVector(n_steps_out))
model.add(LSTM(4096, activation='relu', return_sequences=True, dropout=0.4, recurrent_dropout=0.2))
model.add(TimeDistributed(Dense(n_features)))
model.compile(optimizer='adam', loss='mse')
# fit model
es = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=5, restore_best_weights=True)
model.fit(X, y, batch_size=32, epochs=100, verbose=1, validation_split=0.33, callbacks=[es])
# demonstrate prediction

model.save('data/model.h5')