# multivariate multi-step encoder-decoder lstm example
from numpy import array
from numpy import hstack
from keras.models import load_model
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

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

df = pd.read_csv("data/data.csv", encoding= 'unicode_escape')
df = df.dropna()
df_transactions = df[df['Description'].str.len().gt(3)].dropna().groupby('InvoiceNo')['Description'].agg({'size': len, 'set': lambda x: list(set(x))})

n_steps_in, n_steps_out = 3, 2
n_features = 512

descriptions = np.load('data/descriptions.data.npy')
values = np.load('data/values.data.npy')
clf = NearestCentroid()
clf.fit(values, descriptions)

model = load_model('data/model.h5')

def pad_list(s, n):
    s = [string for string in s if string != ""]
    return [''] * (n - len(s)) + s 

def next(x_in):
    x_input = array([embed(x_in)])
    x_input = x_input.reshape((1, n_steps_in, n_features))
    yhat = model.predict(x_input, verbose=0)
    return list(set(clf.predict(yhat[0,:])))

def predictIndex(i):
    input = pad_list(df_transactions['set'][i], n_steps_in)[:n_steps_in]
    print(input, '=>', next(input))


for i in sample(range(1, 1000), 20):
    predictIndex(i)