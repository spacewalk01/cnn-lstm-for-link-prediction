from __future__ import absolute_import
from __future__ import print_function

from data_utils import vectorize_data, load_paths

from data_utils import display_data, nell_eval, eval_mrr
from data_utils import get_model_answers, get_real_answers
from sklearn import metrics
from sklearn.model_selection import train_test_split
from itertools import chain
from six.moves import range, reduce
import pandas as pd 
import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
import tensorflow as tf
import numpy as np

from sklearn.utils.fixes import signature

from keras import backend as K
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.layers.core import Activation, Dense, Dropout, RepeatVector
from keras.layers import Lambda, Permute, Dropout, add, multiply, dot
from keras.layers import LSTM, Conv1D, GRU, BatchNormalization
from keras.layers import TimeDistributed
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Reshape
from keras.layers import MaxPooling1D, Flatten
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import AveragePooling1D, Bidirectional
from keras import regularizers
from keras.callbacks import LearningRateScheduler
from keras.preprocessing import sequence
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers import concatenate

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
#from keras_pos_embd import TrigPosEmbedding
import math
import sys
import os
np.random.seed(7)

#config = tf.ConfigProto(device_count={"CPU": 20})
#K.tensorflow_backend.set_session(tf.Session(config=config))

config = tf.ConfigProto(intra_op_parallelism_threads=20, inter_op_parallelism_threads=20, \
    allow_soft_placement=True, device_count={'CPU': 20})
session = tf.Session(config=config)
K.set_session(session)

# Parameters/Users/selmee/Documents/charLevelPathEmbedding
#BATCH_SIZE = 64
BATCH_SIZE = 128
NUM_HOPS = 2
#NUM_EPOCHS = 50
NUM_EPOCHS = 50
EMBEDDING_DIM = 100
LSTM_HIDDEN_UNITS = 100
MEMORY_SIZE = 5000
NUM_FILTERS = 50
KERNEL_SIZE = 3
MAX_SENT_LENGTH = 10
LEARNING_RATE = 0.001
NUM_TASKS = 1
POOLING_SIZE = 2
VOCAB_SIZE = 1000
PATIENCE = 30

task = sys.argv[1]

processed_data_dir = "/home/ailab/memory_attention_model/data/processed_data/" 
nell_dir = '/home/ailab/memory_attention_model/data/NELL-995/tasks/concept_'
fb_dir = '/home/ailab/memory_attention_model/data/FB15k-237/tasks/'
kinship_dir = '/home/ailab/memory_attention_model/data/kinship/tasks/'
countries_dir = '/home/ailab/memory_attention_model/data/Countries/tasks/'

task_dir = ''
if sys.argv[2] == "freebase":
    task_dir = fb_dir
    task_uri = '/' + task.replace("@", '/')
elif sys.argv[2] == "nell":
    task_dir = nell_dir
    task_uri = task
elif sys.argv[2] == "kinship":
    task_dir = kinship_dir
    task_uri = task
elif sys.argv[2] == "countries":
    task_dir = countries_dir
    task_uri = "locatedin"

print('task dir:', task_dir)
precision = dict()
recall = dict()
average_precision = dict()
method_names = dict()

#tasks = [
#    'athleteplaysinleague',
#    'worksfor',
#    'organizationhiredperson',
#    'athleteplayssport',
#    'teamplayssport',
#    'personborninlocation',
#    'athletehomestadium',
#    'organizationheadquarteredincity',
#    'athleteplaysforteam']
tasks = [task]

"""
    'agentbelongstoorganization',
    'teamplaysinleague',
    'personleadsorganization'"""

label = ['context', 'e1', 'r', 'e2', 'label']
train = pd.DataFrame(columns=label)
test = pd.DataFrame(columns=label)

for task in tasks:
    print('task name:', task)
    train0, test0 = load_paths(processed_data_dir + task)
    train = pd.concat([train, train0])
    test = pd.concat([test, test0])

#train, test = load_paths(processed_data_dir + task)

def clean_nell(line):
    return line.replace('\n', '').replace('concept:', '').replace('thing$', '').replace("concept_", '')

sort_test = []
train_pos = []
train_neg = []
test_pos = []
test_neg = []

for task in tasks:
    with open(task_dir + task + '/train_pos', 'r') as f: # for learning a single task
        for line in f:
            e1, e2, r = clean_nell(line).lower().split('\t')
            if (e1, r, e2) not in train_pos:
                train_pos.append((e1, r, e2))
                
    with open(task_dir + task + '/train.pairs', 'r') as f:
        for line in f:
            pair, l = clean_nell(line).lower().split(': ')
            e1, e2 = pair.split(',')
            if (e1, task_uri.lower(), e2) not in train_neg and l == '-':
                train_neg.append((e1, task, e2))
                    
    with open(task_dir + task + '/sort_test.pairs', 'r') as f: # for learning a single task
        for line in f:
            pair, l = clean_nell(line).lower().split(': ')
            e1, e2 = pair.split(',')
            e1_uri = e1.lower()
            e2_uri = e2.lower()
            if sys.argv[2] == "freebase":
                e1_uri = '/' + e1.replace('_', '/')
                e2_uri = '/' + e2.replace('_', '/')
            sort_test.append([e1_uri + '' + task_uri, e2_uri, 1 if l == '+' else 0])
            if l == '+':
                test_pos.append((e1_uri, task_uri, e2_uri))
            else:
                test_neg.append((e1_uri, task_uri, e2_uri))

num_tasks = len(tasks)    
print('number of tasks:', num_tasks)

def filter_overlap_train(x):
    sample = (x['e1'], x['r'], x['e2'])
    if x['label'] == '-':        
        return sample not in train_pos
    else:        
        return sample not in train_neg

print('train shape before:', train.shape)
train = train[train.apply(filter_overlap_train, axis=1)]
print('train shape after:', train.shape)

def filter_overlap_test(x):
    sample = (x['e1'], x['r'], x['e2'])
    if x['label'] == '-':
        return sample not in test_pos
    else:        
        return True #if sample not in test_neg 

print('test shape before:', test.shape)
test = test[test.apply(filter_overlap_test, axis=1)]
print('test shape after:', test.shape)        

data = pd.concat([train, test])
vocab = sorted(reduce(lambda x, y: x | y, (set(list(chain.from_iterable(s)) + [r]) for s, r in data[['context', 'r']].values.tolist())))
word2idx = dict((e, i + 1) for i, e in enumerate(vocab))
idx2word = dict((i + 1, e) for i, e in enumerate(vocab))
task2idx = dict((e, i) for i, e in enumerate(tasks))

MAX_SENTS = max(map(len, data['context'].values.tolist()))
MAX_SENT_LENGTH = max(map(len, chain.from_iterable(data['context'].values)))
MEMORY_SIZE = min(MEMORY_SIZE, MAX_SENTS)
VOCAB_SIZE = len(word2idx) + 1 # +1 for nil word

print("Longest sentence length", MAX_SENT_LENGTH)
print("Longest story length", MAX_SENTS)

# split train/validation/test sets
# story, relation, label, char_seq
S, R, L = vectorize_data(train, word2idx, MAX_SENT_LENGTH, MEMORY_SIZE)
trainS, valS, trainR, valR, trainL, valL = train_test_split(S, R, L, test_size=.3, random_state=None)
testS, testR, testL = vectorize_data(test, word2idx, MAX_SENT_LENGTH, MEMORY_SIZE)

# params
n_train = trainS.shape[0]
n_test = testS.shape[0]
n_val = valS.shape[0]

print("Training Size", n_train)
print("Validation Size", n_val)
print("Testing Size", n_test)
print("Entity&Relation Vocab Size", VOCAB_SIZE)

"""def sentEncoder(embedding):
    sent_model = Sequential()
    sent_model.add(embedding)
    sent_model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='same', activation='relu'))
    sent_model.add(Bidirectional(LSTM(LSTM_HIDDEN_UNITS)))
    #sent_model.add(Dropout(0.1))
    return sent_model"""

def sentEncoder(embedding):
    sent_model = Sequential()
    sent_model.add(embedding)
    sent_model.add(Conv1D(NUM_FILTERS, KERNEL_SIZE, padding='same', activation='relu'))
    sent_model.add(MaxPooling1D(pool_size=POOLING_SIZE))
    sent_model.add(Bidirectional(LSTM(int(EMBEDDING_DIM/2))))
    return sent_model

def LSTM_CNN_Encoder(embedding_A):
    sent_model = Sequential()
    sent_model.add(embedding_A)
    sent_model.add(LSTM(int(NUM_FILTERS), return_sequences=True))
    sent_model.add(Conv1D(NUM_FILTERS * 2, KERNEL_SIZE, padding='valid', activation='relu'))
    sent_model.add(MaxPooling1D(pool_size=MAX_SENT_LENGTH - KERNEL_SIZE + 1))
    sent_model.add(Flatten())
    return sent_model

def ConvBiLSTM(step):
    query = Input(shape=(MAX_SENT_LENGTH,))
    context = Input(shape=(MEMORY_SIZE, MAX_SENT_LENGTH,))
    embedding_A = Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAX_SENT_LENGTH, mask_zero=False)

    query_encoded = sentEncoder(embedding_A)(query)
    context_encoded = TimeDistributed(sentEncoder(embedding_A))(context) 
    
    u = query_encoded
    for k in range(step):
        u_rep = RepeatVector(MEMORY_SIZE)(u)
        output = concatenate([context_encoded, u_rep])
        tanh = Dense(EMBEDDING_DIM, activation='tanh')(output)
        score = Dense(1)(tanh)
        score = Flatten()(score)
        alpha = Activation('softmax')(score)
        o = dot([alpha, context_encoded], axes=(1,1))
        u = add([o, u])
        u = Dense(EMBEDDING_DIM, input_shape=(EMBEDDING_DIM,))(u)

    u = Dense(int(EMBEDDING_DIM), activation='relu')(u)
    u = Dense(int(EMBEDDING_DIM/2), activation='relu')(u)

    prediction = Dense(1, activation='sigmoid')(u)
    model = Model([context, query], prediction)
    return model


model = ConvBiLSTM(2)

adam = Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())

trainR1 = sequence.pad_sequences(trainR.reshape([trainR.shape[0], 1]), MAX_SENT_LENGTH)
valR1 = sequence.pad_sequences(valR.reshape([valR.shape[0], 1]), MAX_SENT_LENGTH)
testR1 = sequence.pad_sequences(testR.reshape([testR.shape[0], 1]), MAX_SENT_LENGTH)

#callbacks = [ModelCheckpoint(filepath='output/' + task + '_model.h5', monitor='val_loss', save_best_only=True)]
callbacks = [EarlyStopping(monitor='val_loss', patience=PATIENCE),
                ModelCheckpoint(filepath='output/' + task + '_path4_nell_model.h5', monitor='val_loss', save_best_only=True)]
#callbacks = [ModelCheckpoint(filepath='output/test_model.h5', monitor='val_loss', save_best_only=True)]


history = model.fit([trainS, trainR1], trainL, 
                    validation_data=([valS, valR1], valL),
                    callbacks=callbacks,
                    epochs=NUM_EPOCHS, 
                    batch_size=BATCH_SIZE, verbose=1) 

model.load_weights('output/' + task + '_path4_nell_model.h5')
#model.load_weights('output/joint_model.h5')
#model.load_weights('output/test_model.h5')

evaluation = model.evaluate([testS, testR1], testL, verbose=0)
print('Summary: Loss over the test dataset: %.4f, Accuracy: %.4f' % (evaluation[0], evaluation[1]))

test_preds = model.predict([testS, testR1])
model_answers, real_answers = get_model_answers(test_preds, test.values.tolist())

mean_ap = nell_eval(model_answers, sort_test)                    

mrr, hits_at1, hits_at3, hits_at10 = eval_mrr(model_answers, sort_test)

print("map:", mean_ap)
print("mrr:", mrr)
print("hits_at1:", hits_at1)
print("hits_at3:", hits_at3)
print("hits_at10:", hits_at10)

log_file = sys.argv[3] 

with open(log_file, 'a') as f:
    f.write('relation: ' + task + '\n')
    f.write('MAP:' + str(mean_ap) + '\n')
    f.write('MRR:' + str(mrr) + '\n')
    f.write('hits@1:  ' + str(hits_at1) + '\n')
    f.write('hits@3:  ' + str(hits_at3) + '\n')
    f.write('hits@10: ' + str(hits_at10) + '\n')
    f.write('\n')
                                                
#x = 0
#for task in tasks:
#    correct_answers_task = list(filter(lambda x: task in x[0], sort_test))
#    print('Task:', task)
#    mean_aps = nell_eval(model_answers, correct_answers_task)
#    x += np.mean(mean_aps)
#print('Total MAP:', x/len(tasks))

with open("output/model_answers_path4_" + task + ".txt", 'w') as f:
    for e1, e2, score in model_answers:
        f.write(e1 + ',' + e2 + ',' + str(score) + '\n')