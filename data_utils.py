import os
import pickle
import numpy as np
import json
import sys
import vectorizer

def sortbylength(X, y) :
    len_t = np.argsort([len(x) for x in X])
    X1 = [X[i] for i in len_t]
    y1 = [y[i] for i in len_t]
    return X1, y1
    
def filterbylength(X, y, min_length = None, max_length = None) :
    lens = [len(x)-2 for x in X]
    min_l = min(lens) if min_length is None else min_length
    max_l = max(lens) if max_length is None else max_length

    idx = [i for i in range(len(X)) if len(X[i]) > min_l+2 and len(X[i]) < max_l+2]
    X = [X[i] for i in idx]
    y = [y[i] for i in idx]

    return X, y

def set_balanced_pos_weight(dataset) :
    y = np.array(dataset.train_data.y)
    dataset.pos_weight = [len(y) / sum(y) - 1]

class DataHolder() :
    def __init__(self, X, y, lexicon_feat=None, emoji_feat=None, true_pred=None) :
        self.X = X
        self.y = y
        self.lexicon_feat = lexicon_feat
        self.emoji_feat = emoji_feat
        self.true_pred = true_pred
        self.attributes = ['X', 'y', 'lexicon_feat', 'true_pred']


class Dataset() :
    def __init__(self, name, path, bsize=32, min_length=None, max_length=None) :
        self.name = name
        
        self.vec = pickle.load(open(path, 'rb'))

        X, Xt = self.vec.seq_text['train'], self.vec.seq_text['dev'] # these are lists (of lists) of num. insts-length (NOT PADDED)
        y, yt = self.vec.label['train'], self.vec.label['dev']

        X, y = filterbylength(X, y, min_length=min_length, max_length=max_length)
        Xt, yt = filterbylength(Xt, yt, min_length=min_length, max_length=max_length)
        Xt, yt = sortbylength(Xt, yt)

        self.train_data = DataHolder(X, y)
        self.test_data = DataHolder(Xt, yt)
        
        self.output_size = 1
        self.save_on_metric = 'roc_auc'
        self.keys_to_use = {
            'roc_auc' : 'roc_auc', 
            'pr_auc' : 'pr_auc'
        }

        self.bsize = bsize

