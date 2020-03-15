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

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")

df = pd.read_csv("data/data.csv", encoding= 'unicode_escape')

df = df.dropna()

df_descriptions = df[['Description']].dropna().drop_duplicates()
df_descriptions = df_descriptions[df_descriptions['Description'].str.len().gt(3)]

descriptions = df_descriptions['Description'].tolist()
values = embed(descriptions)
np.save('data/descriptions.data', descriptions)
np.save('data/values.data', values)
clf = NearestCentroid()
clf.fit(values, descriptions)

# print(clf.predict(embed(["toast"])))

df_transactions = df[df['Description'].str.len().gt(3)].dropna().groupby('InvoiceNo')['Description'].agg({'size': len, 'set': lambda x: list(set(x))})

def pad_list(s, n):
    s = [string for string in s if string != ""]
    return [''] * (n - len(s)) + s 

def find_subsets(s, n):
    m = min(len(s), n)
    return list(itertools.permutations(s, m))

def random_find_subsets(s, n, m):
    l = find_subsets(s, n)
    return sample(l, m)

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    X, y = list(), list()
    for seq in sequences:
        if len(seq) >= 3 and len(seq) <= 6:
            for inner_seq in random_find_subsets(seq, n_steps_in+n_steps_out, 3):
                padddes_seq = pad_list(inner_seq, n_steps_in+n_steps_out)
                seq_x, seq_y = padddes_seq[:n_steps_in], padddes_seq[-n_steps_out:]
                X.append(embed(seq_x))
                y.append(embed(seq_y))
    return array(X), array(y)


# choose a number of time steps
dataset = df_transactions['set'].tolist()
print(len(dataset))

n_steps_in, n_steps_out = 3, 2
# covert into input/output
X, y = split_sequences(dataset, n_steps_in, n_steps_out)
print(X.shape)

np.save('data/X.data', X)
np.save('data/y.data', y)