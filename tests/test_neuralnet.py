#!/usr/bin/env python

'''
Example of use of abnet2
'''

from glob import glob
import os.path as path
import time
import pickle
import re

import numpy as np
import lasagne
import theano.tensor as T
import theano

import abnet2


def read_alignment(alignment_file='data/alignment.txt'):
    ''' read the alignement file in format ... 
    
    the function will return 
    
    labels: dictionary with the phoneme as a key
            and the values the element number
    
    alignments: list 
    '''
    labels = set()
    alignments = {}
    with open(alignment_file) as fin:
        for line in fin:
            splitted = line.split()
            alignment = [re.split('[0-9]', phn.split('_')[0])[0] for phn in splitted[1:]]
            labels.update(alignment)
            alignments[splitted[0]] = alignment
    labels = {l: i for i, l in enumerate(labels)}
    return labels, alignments


def stack_fbanks(features, n):
    '''stack filter banks'''
    assert n % 2 == 1, 'number of stacked frames must be odd'
    dim = features.shape[1]
    pad = np.zeros((n/2, dim), dtype=features.dtype)
    features = np.concatenate((pad, features, pad))
    aux = np.array([features[i:i-n+1]
                    for i in xrange(n-1)] + [features[n-1:]])
    return np.reshape(np.swapaxes(aux, 0, 1), (-1, dim * n))

def load_data():
    '''load the alignment file and the precomputed filterbanks for this experiment'''
    labels, alignments = read_alignment()
    y = []
    fbanks = np.load('data/fbanks.npz')
    stacked_fbanks = []
    for f in fbanks:
        stacked_fbanks.append(stack_fbanks(fbanks[f], 5))
        # stacked_fbanks.append(fbanks[f])
        nframes = len(alignments[f])
        aux = np.zeros((nframes), dtype='int32')
        for i_frame, phone in enumerate(alignments[f]):
            aux[i_frame] = labels[phone]
        y.append(aux)
    X = np.concatenate(stacked_fbanks, axis=0)
    y = np.concatenate(y, axis=0)
    X = (X - np.mean(X)) / np.std(X)
    print('shape X = {} \t \t shape y = {}'.format(X.shape, y.shape))
    return X, y

def mdelta(X):
    ''' split data in train and test datatsets '''
    # TODO: use instead sklearn train_test_split, see http://goo.gl/haLYso 
    X_train1, X_train2, X_val1, X_val2 = np.vstack((X[:-5001], X[:-5100])), np.vstack((X[1:-5000], X[100:-5000])), np.vstack((X[-5000:-1], X[-5000:-100])), np.vstack((X[-4999:], X[-4900:]))
    y_train, y_val = np.concatenate((np.ones((X.shape[0]-5001,), dtype='int32'), np.zeros((X.shape[0]-5100,), dtype='int32'))), np.concatenate((np.ones((4999), dtype='int32'), np.zeros((4900,), dtype='int32')))

    return X_train1, X_train2, y_train, X_val1, X_val2, y_val


def test_phone_classifier():
    '''Test abnet2 '''
    X, y = load_data()
    print('max_y={}'.format(y.max()+1))
    nnet = abnet2.Classifier_Nnet([X.shape[1], 200, np.max(y)+1], [0.1, 0.2])
    X_train, X_val = X[:-5000], X[5000:]
    y_train, y_val = y[:-5000], y[5000:]
    nnet.train(X_train, y_train, X_val, y_val, max_epochs=2000, patience=20)


if __name__ == '__main__':
    # test_phone_classifier()

    X, _ = load_data()
    X_train1, X_train2, y_train, X_val1, X_val2, y_val = mdelta(X)
    nnet = abnet2.ABnet([200, 200, 58])

    print("Starting training...")
    nnet.train(X_train1, X_train2, y_train, X_val1, X_val2, y_val, max_epochs=100, patience=5)
    # with open('abnet.pickle', 'w') as fout:
    #     pickle.dump(nnet, fout)
    # with open('abnet.pickle') as fin:
    #     nnet = pickle.load(fin)

    for f in X:
        print np.shape(X[f])
        embs = nnet.evaluate(X[f])
        np.save('embs/'+f, embs)

