# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:04 2015

@author: thalita
"""
import numpy as np
import pickle as pkl
from base import SavedRecommendations
from datasplit import Split
from utils import to_gzpickle, read_gzpickle


MF_SUFFIX = '_mf.pkl'
TRAIN_SUFFIX = '_train.pkl'
TEST_SUFFIX = '_rec.pkl'


def load_split(split_fname_prefix, fold=None):
    if fold is None:
        fname =  split_fname_prefix + '_split.pkl'
    else:
        fname =  split_fname_prefix + '_split_%d.pkl' % fold

    split = read_gzpickle(fname)
    split.train = MatrixDatabase(split.train.get_matrix())
    to_gzpickle(split, fname)
    return split


def gen_mf(split, filepath, RS):
    matrices = RS.gen_mf(split.train)
    fname = filepath + MF_SUFFIX
    to_gzpickle(matrices, fname)


def load_mf(filepath, RS):
    fname = filepath + MF_SUFFIX
    matrices = read_gzpickle(fname)
    RS.load_mf(*matrices)
    return RS


def train_save(RS, split, out_filepath):
    RS.fit(split.train)
    RS.save(out_filepath+TRAIN_SUFFIX)


def test_save(RS, out_filepath, split):
    RS.load(out_filepath+TRAIN_SUFFIX, split.train)
    rec = SavedRecommendations()
    rec.save(out_filepath+TEST_SUFFIX, RS)


def ensemble_train_save(ens, out_filepath, split):
    ens.fit(split)
    ens.save(out_filepath+TRAIN_SUFFIX)


def ensemble_test_save(ens, out_filepath, split):
    ens.load(out_filepath+TRAIN_SUFFIX, split.train)
    rec = SavedRecommendations()
    rec.save(out_filepath+TEST_SUFFIX, ens)


def load_model(RS, out_filepath, split):
    RS.load(out_filepath+TRAIN_SUFFIX, split.train)


def load_recommendations(filepath):
    rec = SavedRecommendations()
    rec.load(filepath+TEST_SUFFIX)
    return rec


class Metrics(object):
    __atN__ = [1, 5, 10, 15, 20]
    def ir_metric_names(which, atNs=None):
        if atNs is None:
            atNs = Metrics.__atN__
        metrics = ['P@%d_' % atN + which for atN in atNs]
        metrics += ['R@%d_' % atN + which for atN in atNs]
        metrics += ['F1@%d_' % atN + which for atN in atNs]
        return metrics

    def error_metric_names(which, user=False):
        return ['RMSE'+('u' if user else '')+ '_' + which,
                'MAE'+('u' if user else '') + '_' + which]

    def ensemble_metrics_names():
        return ['kendalltau', 'stddev']

    def __init__(self, split, filepath=None, RS=None):
        self.RS = SavedRecommendations()
        if RS is not None:
            self.RS = RS
        elif filepath is not None:
            self.RS.load(filepath+TEST_SUFFIX)
        else:
            raise ValueError('Must inform either path to recommender\
            or a recommender object')
        self.split = split
        self.test_set = None
        self.which = None
        self.metrics = dict()

    def _rlist_single_user(self, user_id, threshold):
        hidden_items = [i_id for i_id, rating in self.test_set[user_id]]
        candidate_items = list(self.split.train.get_unrated_items(user_id)) \
            + hidden_items

        rlist = self.RS.recommend(user_id,
                                  threshold=threshold,
                                  candidate_items=candidate_items)
        return rlist

    def _hits_atN(self, user_id, rlist, atN, threshold):
        good_hidden = [i_id for i_id, rating in self.test_set[user_id]
                       if rating > threshold]
        rlist = dict(rlist[0:atN])
        hit = 0
        for item_id in good_hidden:
            hit += 1 if item_id in rlist else 0
        return hit

    def def_test_set(self, which):
        self.which = which
        if which == 'test':
            self.test_set = self.split.test
        elif which == 'valid':
            self.test_set = self.split.valid
        else:
            raise ValueError("Invalid set name: %s (user 'valid' or 'test')"
                             % which)

    def list_metrics(self, threshold):
        if self.test_set is None:
            raise ValueError('def_test_set must be called before metrics \
            computation')
        recall = 0
        precision = 0
        F1 = 0
        n_users = len(self.test_set)
        for user_id in self.test_set:
            rlist = self._rlist_single_user(user_id, threshold)
            for atN in Metrics.__atN__:
                hits = self._hits_atN(user_id, rlist, atN, threshold)
                r = hits/len(self.test_set[user_id])
                p = hits/atN
                if r+p > 0:
                    f1 = 2*r*p/(r+p)
                else:
                    f1 = 0
                recall += r/n_users
                precision += p/n_users
                F1 += f1/n_users

                self.metrics['P@%d_' % atN + self.which] = precision
                self.metrics['R@%d_' % atN + self.which] = recall
                self.metrics['F1@%d_' % atN + self.which] = F1

    def _absErr_single_rating(self, user_id, item_id, true_rating):
        pred_rating = self.RS.predict(user_id, item_id)
        absErr = np.abs(pred_rating - true_rating)
        return absErr

    def error_metrics(self):
        if self.test_set is None:
            raise ValueError('def_test_set must be called before metrics \
            computation')
        MAE = 0
        MSE = 0
        MAEu = 0
        MSEu = 0
        nUsers = len(self.test_set)
        nTestRatings = sum([len(r) for r in self.test_set.values()])
        for user, test in self.test_set.items():
            for item, rating in test:
                absErr = self._absErr_single_rating(user, item, rating)
                MAE += absErr/nTestRatings
                MSE += absErr**2/nTestRatings
                MAEu += absErr/len(test)
                MSEu += absErr**2/len(test)
        MAEu /= nUsers
        MSEu /= nUsers
        RMSE = np.sqrt(MSE)
        RMSEu = np.sqrt(MSEu)
        self.metrics['RMSE_' + self.which] = RMSE
        self.metrics['MAE_' + self.which] = MAE
        self.metrics['RMSEu_' + self.which] = RMSEu
        self.metrics['MAEu_' + self.which] = MAEu

    def ensemble_metrics(self):
        if '_kendalltau_avg' in self.RS.config:
            self.metrics['kendalltau'] = np.mean(self.RS.config['_kendalltau_avg'])
            print('K: ', self.metrics['kendalltau'])
        if '_stddev_avg' in self.RS.config:
            self.metrics['stddev'] = np.mean(self.RS.config['_stddev_avg'])
            print(self.metrics['stddev'])

