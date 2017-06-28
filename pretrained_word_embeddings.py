"""
This file will use pretrained word embeddings to prepare representations
of the argument and the wikipedia articles, then do a sum over the dot
product of the representation of the argument with each wikipedia article
and finally use the sum as the representation of argument combined with
knowledge from wikipedia. This final combined representations of two
articles will be used to do classification between the two for which one
is more convincing using logistic regression.
"""

'''This script loads pre-trained word embeddings (GloVe embeddings)
into a frozen Keras Embedding layer, and uses it to
train a text classification model on the 20 Newsgroup dataset
(classication of newsgroup messages into 20 different categories).
GloVe embedding data can be found at:
http://nlp.stanford.edu/data/glove.6B.zip
(source page: http://nlp.stanford.edu/projects/glove/)
20 Newsgroup data can be found at:
http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.html
'''

from __future__ import print_function

import csv
from collections import Counter, defaultdict
import math
from os import listdir
from scipy.stats import pearsonr, spearmanr
from numpy import mean
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
import random

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model

BASE_DIR = './data/'

GLOVE_DIR = BASE_DIR + 'glove/'
ARGUMENT_DATA_DIR = BASE_DIR + 'arguments/'
WIKI_DATA_DIR = BASE_DIR + 'wiki/'

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
VALIDATION_SPLIT = 0.2

# first, build index mapping words in the embeddings set
# to their embedding vector

print('Indexing word vectors.')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

# second, prepare text samples and their labels
print('Processing arguments dataset')

pid = []
argument1 = []
argument2 = []
labels_ = []


for csv_file in sorted(os.listdir(ARGUMENT_DATA_DIR)):
    """
    with open(ARGUMENT_DATA_DIR + csv_file, 'rb') as f:
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            _, _, text1, text2 = row
            all_arguments += [text1, text2]
    """
    with open(ARGUMENT_DATA_DIR + csv_file, 'rb') as f:
        preds = []
        labels = []
        reader = csv.reader(f, delimiter='\t')
        next(reader, None)
        for row in reader:
            pid, label, text1, text2 = row
            labels.append(int(label[1]) - 1)
            texts = [text1, text2] #+ [w for w in wiki if len(w) < 2000]

            cv = CountVectorizer(stop_words='english', binary=False)
            vecs = cv.fit_transform(texts)
            words = vecs.get_feature_names()

            if embeddings_index[words[0]] == None:
                V_d = embeddings_index[words['unk']]
            else:
                V_d = embeddings_index[words[0]]

            for word in words[1:]:
                if embeddings_index[word] not None:
                    V_d = np.vstack([V_d, embeddings_index[word]])
                else:
                    V_d = np.vstack([V_d, embeddings_index['unk']])

            cur_max1 = np.dot(vecs[0,:], V_d).sum()#np.amax(np.dot(vecs[0,:], vecs[2:,:].T))
            cur_max2 = np.dot(vecs[1,:], V_d).sum()#np.amax(np.dot(vecs[1,:], vecs[2:,:].T))

            print "score of arg 1"
            print cur_max1

            print "score of arg 2"
            print cur_max2

            if cur_max1 == cur_max2:
                preds.append(bool(random.randint(0,1)))
            else:
                preds.append( cur_max2>cur_max1 )
        acc = accuracy_score(labels, preds)
        print csv_file,acc
        acc_scores.append(acc)
print mean

"""
texts = []  # list of text samples
labels_index = {}  # dictionary mapping label name to numeric id
labels = []  # list of label ids
for name in sorted(os.listdir(ARGUMENT_DATA_DIR)):
    path = os.path.join(ARGUMENT_DATA_DIR, name)
    if os.path.isdir(path):
        label_id = len(labels_index)
        labels_index[name] = label_id
        for fname in sorted(os.listdir(path)):
            if fname.isdigit():
                fpath = os.path.join(path, fname)
                if sys.version_info < (3,):
                    f = open(fpath)
                else:
                    f = open(fpath, encoding='latin-1')
                t = f.read()
                i = t.find('\n\n')  # skip header
                if 0 < i:
                    t = t[i:]
                texts.append(t)
                f.close()
                labels.append(label_id)

print('Found %s texts.' % len(texts))


# finally, vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]
num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-num_validation_samples]
y_train = labels[:-num_validation_samples]
x_val = data[-num_validation_samples:]
y_val = labels[-num_validation_samples:]

print('Preparing embedding matrix.')

# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index))
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

# load pre-trained word embeddings into an Embedding layer
# note that we set trainable = False so as to keep the embeddings fixed
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)

print('Training model.')

# train a 1D convnet with global maxpooling
sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
embedded_sequences = embedding_layer(sequence_input)
x = Conv1D(128, 5, activation='relu')(embedded_sequences)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(5)(x)
x = Conv1D(128, 5, activation='relu')(x)
x = MaxPooling1D(35)(x)
x = Flatten()(x)
x = Dense(128, activation='relu')(x)
preds = Dense(len(labels_index), activation='softmax')(x)

model = Model(sequence_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])

model.fit(x_train, y_train,
          batch_size=128,
          epochs=10,
          validation_data=(x_val, y_val))
"""
