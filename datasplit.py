# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:33:50 2015

@author: thalita
"""
import abc
import pickle as pkl
import numpy as np
from sklearn.cross_validation import KFold
from databases import HiddenRatingsDatabase
from collections import defaultdict


class Split(object):
    def __init__(self, train, valid=None, test=None, config=None):
        self.train = train
        self.valid = valid
        self.test = test
        self.config = config


class Splitter(object):
    __metaclass__ = abc.ABCMeta

    @property
    def per_user(self):
        return self._per_user

    @per_user.setter
    def per_user(self, val):
        self._per_user = val

    @property
    def train(self):
        return self._train

    @train.setter
    def train(self, val):
        self._train = val

    @property
    def test(self):
        return self._test

    @test.setter
    def test(self, val):
        self._test = val

    @abc.abstractmethod
    def split(self, database):
        pass

    def save(self, filepath):
        fname_prefix = filepath + self.suffix
        if self.nfolds == 1:
            split = Split(train=self.train, test=self.test, config=self.config)
            with open(fname_prefix + '_split.pkl', 'wb') as f:
                pkl.dump(split, f)
        else:
            for i in range(self.nfolds):
                config = self.config.copy()
                config['fold'] = i
                split = Split(train=self.train[i], test=self.test[i], config=config)
                fname = fname_prefix + '_split_%d.pkl' % i
                with open(fname, 'wb') as f:
                    pkl.dump(split, f)
        return fname_prefix


class CVTestRatingSplitter(Splitter):
    def __init__(self, nfolds=5, per_user=True, pct_hidden=0.2, threshold=0):
        self.nfolds = nfolds
        self.per_user = per_user
        self.pct_hidden = pct_hidden
        self.config = {'nfolds': nfolds, 'pct_hidden': pct_hidden,
                       'per_user': per_user, 'threshold': threshold}
        if nfolds <= 1:
            self.CV_splitter = HoldoutRatingSplitter(pct_hidden, per_user,
                                                    threshold)
        else:
            self.CV_splitter = kFoldRatingSplitter(nfolds, per_user)

        self.Test_splitter = HoldoutRatingSplitter(pct_hidden, per_user,
                                                   threshold)
        self.train = None
        self.test = None
        self.valid = None
        self.suffix = '_CV' + self.CV_splitter.suffix \
            + '_test' + self.Test_splitter.suffix

    def split(self, database):
        self.Test_splitter.split(database)
        self.CV_splitter.split(self.Test_splitter.train)
        self.train = self.CV_splitter.train
        self.valid = self.CV_splitter.test
        self.test = self.Test_splitter.test

    def save(self, filepath):
        fname_prefix = filepath + self.suffix
        if self.nfolds == 1:
            split = Split(self.train, self.valid, self.test, self.config)
            with open(fname_prefix + '_split.pkl', 'wb') as f:
                pkl.dump(split, f)
        else:
            for i in range(self.nfolds):
                config = self.config.copy()
                config['fold'] = i
                split = Split(self.train[i], self.valid[i], self.test, config)
                fname = fname_prefix + '_split_%d.pkl' % i
                with open(fname, 'wb') as f:
                    pkl.dump(split, f)
        return fname_prefix

class kFoldRatingSplitter(Splitter):
    def __init__(self, nfolds=5, per_user=True):
        self.per_user = per_user
        self.nfolds = nfolds
        self.suffix = '_%dfold' % self.nfolds
        self.config = {'nfolds': nfolds, 'per_user': per_user}
        self.hidden_coord = [[] for i in range(nfolds)]
        self.train = None
        self.test = None

    def split(self, database):
        if self.per_user:
            for u in range(database.n_users()):
                item_ratings = database.get_rating_list(u)
                size = len(item_ratings)
                splits = list(KFold(size, n_folds=self.nfolds, shuffle=True))
                #splits are in format [...,(train_i,test_i),...]
                for i in range(self.nfolds):
                    for idx in splits[i][1]:
                        self.hidden_coord[i].append((u, item_ratings[idx][0]))
        else:
            raise NotImplementedError()

        self.train = \
            [HiddenRatingsDatabase(database.get_matrix(), split)
             for split in self.hidden_coord]

        self.test = []
        for split in self.hidden_coord:
            users = defaultdict(list)
            for u, i in split:
                r = database.get_matrix()[u, i]
                users[u].append((i, r))
            self.test.append(users)


class HoldoutRatingSplitter(Splitter):
    def __init__(self, pct_hidden=0.2, per_user=True, threshold=0):
        self.per_user = per_user
        self.pct_hidden = pct_hidden
        self.threshold = threshold
        self.nfolds = 1
        self.suffix = '_%d_%d_holdout' % (int((1-pct_hidden)*100),
                                         int(pct_hidden*100))
        self.config = {'nfolds': self.nfolds, 'per_user': per_user,
                       'pct_hidden': pct_hidden, 'threshold': threshold}
        self.hidden_coord = []
        self.train = None
        self.test = None

    def _get_hidden(self, matrix):
        # get positions equal to or above threshold (ratings)
        row, col = np.where(matrix >= self.threshold)
        # len(row)== total number of ratings>=threshold
        n_hidden = np.ceil(self.pct_hidden*len(row))
        # pick n_hidden random positions
        hidden_idx = np.random.randint(0, len(row), n_hidden)
        return (row[hidden_idx], col[hidden_idx])

    def split(self, database):
        matrix = np.array(database.get_matrix())
        if self.per_user:
            for u, user_vector in enumerate(matrix):
                rows, cols = self._get_hidden(np.array(user_vector, ndmin=2))
                self.hidden_coord += \
                    list(zip([u for _ in range(len(rows))], cols))
        else:
            rows, cols = self._get_hidden(matrix)
            self.hidden_coord = list(zip(rows, cols))

        self.train = \
            HiddenRatingsDatabase(database.get_matrix(), self.hidden_coord)

        self.test = defaultdict(list)
        for u, i in self.hidden_coord:
            r = database.get_matrix()[u, i]
            self.test[u].append((i, r))

