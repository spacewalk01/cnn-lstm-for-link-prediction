from __future__ import absolute_import
from __future__ import division

import os
import re
import numpy as np
import csv
from collections import defaultdict
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import itertools
import pandas as pd 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def find_common_range_of_relation(target):
    word_list = []
    for e1, r, e2 in kb:
        if r in target:
            for e in e2.split('_'):
                if e not in stop_words:
                    word_list.append(e)
            
    dict = {}
    for word in word_list:
        if len(word) > 0:
            dict[word] = dict.get(word, 0) + 1
    type_counts = sorted(dict.items(), key = lambda x: x[1], reverse = True)
    common_types = list(filter(lambda x:x[1] > 50 and len(x[0]) > 1, type_counts))[:10]
    return list(map(lambda x:x[0], common_types))

def find_common_domain_of_relation(target):
    word_list = []
    for e1, r, e2 in kb:
        if r in target: 
            for e in e1.split('_'):
                if e not in stop_words:
                    word_list.append(e)
            
    dict = {}
    for word in word_list:
        if len(word) > 0:
            dict[word] = dict.get(word, 0) + 1
    type_counts = sorted(dict.items(), key = lambda x: x[1], reverse = True)
    common_types = list(filter(lambda x:x[1] > 80 and len(x[0]) > 1, type_counts))[:10]
    return list(map(lambda x:x[0], common_types))

def get_model_answers(preds, data):
    model_answers = []
    real_answers = []
    i = 0
    for _, e1, r, e2, l in data:
        q = e1 + r
        score = preds[i][0]
        model_answers.append([q, e2, score])
        y = 1 if l[0] == '+' else 0
        real_answers.append([q, e2, y])
        i += 1
    return model_answers, real_answers

def get_real_answers(filename):
    with open(filename) as f:
        test_data = []
        for line in f.readlines():
            e1, rest = line.replace("thing$concept_", "").split(",")
            e2, sign = rest.split(":")
            sign = sign.replace("\n", "").strip()
            label = 0
            if sign == "+":
                label = 1
            test_data.append([e1, e2, label])
        return test_data

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def nell_eval(model_answers, correct_answers):
    test_data = correct_answers

    # load prediction scores
    preds = {}
    for line in model_answers:
        e1, e2, score = line
        score = float(score)
        if (e1, e2) not in preds:
            preds[(e1, e2)] = score
        else:
            if preds[(e1, e2)] < score:
                preds[(e1, e2)] = score

    def get_pred_score(e1, e2):
        if (e1, e2) in preds:
            return preds[(e1, e2)]
        else:
            return -np.inf
    test_pairs = defaultdict(lambda : defaultdict(int))
    for e1, e2, label in test_data:
        test_pairs[e1][e2] = label
    aps = []
    score_all = []

    # calculate MAP
    for e1 in test_pairs:
        y_true = []
        y_score = []
        for  e2 in test_pairs[e1]:
            score = get_pred_score(e1, e2)
            score_all.append(score)
            y_score.append(score)
            y_true.append(test_pairs[e1][e2])
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)

        ranks = []
        correct = 0
        for idx_, item in enumerate(count):
            if item[1] == 1:
                correct += 1
                ranks.append(correct / (1.0 + idx_))
        
        if len(ranks) == 0:
            ranks.append(0)
        aps.append(np.mean(ranks))
    mean_ap = np.mean(aps)
    print('{0} queries evaluated'.format(len(aps)))
    return mean_ap

from collections import defaultdict
def PR_curve(model_answers, correct_answers):
    test_data = correct_answers

    # load prediction scores
    preds = {}
    for line in model_answers:
        e1, e2, score = line
        score = float(score)
        if (e1, e2) not in preds:
            preds[(e1, e2)] = score
        else:
            if preds[(e1, e2)] < score:
                preds[(e1, e2)] = score

    def get_pred_score(e1, e2):
        if (e1, e2) in preds:
            return preds[(e1, e2)]
        else:
            return -np.inf
    test_pairs = defaultdict(lambda : defaultdict(int))
    for e1, e2, label in test_data:
        test_pairs[e1][e2] = label
    aps = []
    score_all = []

    # calculate MAP
    
    precisions = []
    recalls = []
    for e1 in test_pairs:
        y_true = []
        y_score = []
        for  e2 in test_pairs[e1]:
            score = get_pred_score(e1, e2)
            score_all.append(score)
            y_score.append(score)
            y_true.append(test_pairs[e1][e2])
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)

        correct = 0
        found = 0
        prec = []
        rec = []

        num_corrects = len(list(filter(lambda x: x[1], count)))

        for idx_, item in enumerate(count):
            if item[1] == 1: 
                correct += 1
                prec.append(correct / (1.0 + idx_))
                rec.append(correct / num_corrects)
                found = 1
        precisions.append(sum(prec)/len(prec))
        recalls.append(sum(rec)/len(rec))
        
    return [precisions, recalls, np.mean(np.array(precisions))]
    
def eval_mrr(model_answers, correct_answers):
    test_data = correct_answers

    # load prediction scores
    preds = {}
    for line in model_answers:
        e1, e2, score = line
        score = float(score)
        if (e1, e2) not in preds:
            preds[(e1, e2)] = score
        else:
            if preds[(e1,e2)] < score:
                preds[(e1,e2)] = score

    def get_pred_score(e1, e2):
        if (e1, e2) in preds:
            return preds[(e1,e2)]
        else:
            return -np.inf
    test_pairs = defaultdict(lambda : defaultdict(int))
    for e1, e2, label in test_data:
        test_pairs[e1][e2] = label
    mrr_ranks = []
    score_all = []
    hits_at1 = []
    hits_at3 = []
    hits_at10 = []
    ranks_list = []
    # calculate MRR
    for e1 in test_pairs:
        query_ranks = []
        y_true = []
        y_score = []
        for e2 in test_pairs[e1]:
            score = get_pred_score(e1, e2)
            score_all.append(score)
            y_score.append(score)
            y_true.append(test_pairs[e1][e2])
        count = list(zip(y_score, y_true))
        count.sort(key=lambda x: x[0], reverse=True)
        mrr_rank = 0.0
        found_rank = 0.0
        answer_rank = []
        found = 0
        for idx_, item in enumerate(count):
            if item[1] == 1 and found_rank == 0:
                mrr_rank = 1.0 / (1.0 + idx_)                
                found_rank = 1
            if item[1] == 1 and found == 0:
                answer_rank.append(idx_ + 1.0)
                found = 1
        mrr_ranks.append(mrr_rank)
        ranks_list.append(answer_rank)

    hits_at1 = np.sum( [ (np.sum(np.array(ranks) <= 1)) for ranks in ranks_list] )/len(test_pairs)
    hits_at3 = np.sum( [ (np.sum(np.array(ranks) <= 3)) for ranks in ranks_list] )/len(test_pairs)
    hits_at10= np.sum( [ (np.sum(np.array(ranks) <= 10)) for ranks in ranks_list] )/len(test_pairs)

    mrr = np.mean(mrr_ranks)
    return (mrr, hits_at1, hits_at3, hits_at10)

def print_epoch(epoch, num_epoch, train_loss, val_loss, train_acc, val_acc):
    print("Epoch: {0:2d}/{1}\tloss: {2: 0.7f}\tacc: {3: 0.4f}\tval_loss: {4: 0.7f}\tval_acc: {5: 0.4f}".format(epoch, 
        num_epoch, train_loss, train_acc, val_loss, val_acc))

def display_data(data):
    story = data[0]
    question = data[1]
    answer = data[2]
    print("Story:")
    for line in story:
        print(" ".join(line))
    print("Question:")
    print(" ".join(question) + "?")
    print("Answer:", answer)

def load_paths(data_dir, only_supporting=False):
    '''Load the nth task. There are 20 tasks in total.

    Returns a tuple containing the training and testing data for the task.
    '''  
    files = os.listdir(data_dir)
    files = [os.path.join(data_dir, f) for f in files]
    train_file = [f for f in files if 'train.txt' in f][0]
    test_file = [f for f in files if 'test.txt' in f][0]
    train_data = get_stories(train_file, only_supporting)
    test_data = get_stories(test_file, only_supporting)
    return train_data, test_data

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\S+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbI tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    nid = 0
    for line in lines:
        line = str.lower(line).replace("\n", "")
        if nid == 1:
            story = []
        if '\t' in line: # query
            nid = 1
            e1, r, e2, l = line.split('\t')
            substory = [x for x in story if x]
            data.append((substory, e1, r, e2, l))
            story = []
        else: # regular sentence
            # remove periods
            nid = 0
            sent = tokenize(line)
            if sent[-1] == ".":
                sent = sent[:-1]
            story.append(sent)
    label = ['context', 'e1', 'r', 'e2', 'label']
    df = pd.DataFrame.from_records(data, columns=label)            
    return df

def get_stories(f, only_supporting=False):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    with open(f) as f:
        return parse_stories(f.readlines(), only_supporting=only_supporting)

def vectorize_data(data, word_idx, SENT_SIZE, MEMORY_SIZE):
    """
    Vectorize stories and queries.

    If a sentence length < sentence_size, the sentence will be padded with 0's.

    If a story length < memory_size, the story will be padded with empty memories.
    Empty memories are 1-D arrays of length sentence_size filled with 0's.

    The answer array is returned as a one-hot encoding.
    """
    S = []
    R = []
    L = []

    for story, _, relation, _, answer in data.values.tolist():
        cxt = []
        for i, path in enumerate(story, 1):
            ls = max(0, SENT_SIZE - len(path))
            cxt.append([0] * ls + [word_idx[w] for w in path])

        # pad to memory_size
        for _ in range(max(0, MEMORY_SIZE - len(cxt))):
            cxt.append([0] * SENT_SIZE)

        S.append(cxt)
 #       C.append(char_cxt)
        R.append(word_idx[relation])
        L.append(1 if answer[0] == '+' else 0)

    return np.array(S), np.array(R), np.array(L)