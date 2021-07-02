# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 16:00:05 2019

@author: prajw
"""

#import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Concatenate, Dense
from keras.optimizers import Adam
import numpy as np
import pickle

with open("embeddings_1.pickle", "rb+") as inpfile:
    embeddings = pickle.load(inpfile)

qids, embeddings = zip(*embeddings)
qids = np.array(qids)
embeddings = np.array(embeddings)

n = len(qids)

mapping = {}
for i in range(n):
    mapping[qids[i]] = embeddings[i]

sentence1, sentence2 = [], []
y = []
with open("question_pairs.txt", "r+") as inpfile:
    for line in inpfile:
        l = [int(x) for x in line.strip().split(" ")]
        if l[0] < n and l[1] < n:
            sentence1.append(mapping[qids[l[0]]])
            sentence2.append(mapping[qids[l[1]]])
            y.append(l[2])
            
del qids
del embeddings
del mapping

sentence1 = np.array(sentence1)
sentence2 = np.array(sentence2)
y = np.array(y)

#sentence1 = tf.convert_to_tensor(sentence1)
#sentence2 = tf.convert_to_tensor(sentence2)

input1 = Input(shape=(4096,))
input2 = Input(shape=(4096,))

cos = Concatenate()([input1, input2])

dense1 = Dense(8192, activation='relu')(cos)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=[input1, input2], outputs=dense3)


model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-3, decay=1e-3/200), metrics=["binary_accuracy"])

sentence1_train = sentence1[:35000]
sentence2_train = sentence2[:35000]
y_train = y[:35000]

sentence1_test = sentence1[35000:]
sentence2_test = sentence2[35000:]
y_test = y[35000:]

model.fit([sentence1_train, sentence2_train], y_train, epochs=20, batch_size=64)
