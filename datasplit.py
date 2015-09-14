# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 12:33:50 2015

@author: thalita
"""
import abc
import pickle as pkl
import numpy as np
from sklearn.cross_validation import KFold
from database import HiddenRatingsDatabase


class Split(object):
    def _init_(self, train, test, config):
        self.train = train
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
        self.config = self.__dict__
        split = Split(self.train, self.test, self.config)
        with open(filepath + self.suffix + '_split.pkl', 'wb') as f:
            pkl.dump(split, f)


class kFoldRatingSplitter(Splitter):
    def __init__(self, nfolds=5, per_user=True):
        self.per_user = per_user
        self.nfolds = nfolds
        self.hidden_coord = [[] for i in range(nfolds)]
        self.train = None
        self.test = None
        self.config = None
        self.suffix = '_%dfold' % self.nfolds


    def split(self, database):
        self.shape = database.get_matrix().shape
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
        # saved in order :(rating, user, item)
        self.test = \
            [[(database.get_matrix()[u, i], u, i) for u, i in split]
             for split in self.hidden_coord]


class HoldoutRatingSplitter(Splitter):
    def __init__(self, pct_hidden=0.2, per_user=True, threshold=0):
        self.per_user = per_user
        self.pct_hidden = pct_hidden
        self.threshold = threshold
        self.hidden_coord = []
        self.train = None
        self.test = None
        self.config = None
        self.nfolds = 1
        self.suffix = '%d_%d_holdout' % (int((1-pct_hidden)*100),
                                                   int(pct_hidden*100))

    def _get_hidden(self, matrix):
        # get positions equal to or above threshold (ratings)
        row, col = np.where(matrix >= self.threshold)
        # len(row)== total number of ratings>=threshold
        n_hidden = np.ceil(self, self.pct_hidden*len(row))
        # pick n_hidden random positions
        hidden_idx = np.random.randint(0, len(row), n_hidden)
        return (row[hidden_idx], col[hidden_idx])

    def split(self, database):

        matrix = np.array(database.get_matrix())
        self.shape = matrix.shape
        if self.per_user:
            for u, user_vector in enumerate(matrix):
                rows, cols = self._get_hidden(user_vector)
                self.hidden_coord += list(zip(rows, cols))
        else:
            rows, cols = self._get_hidden(matrix)
            self.hidden_coord =  list(zip(rows, cols))

        self.train = \
            HiddenRatingsDatabase(database.get_matrix(), self.hidden_coord)
        # saved in order :(rating, user, item)
        self.test = \
            [(database.get_matrix()[u, i], u, i) for u, i in self.hidden_coord]

