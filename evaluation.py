# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:04 2015

@author: thalita
"""
import numpy as np
import pickle as pkl
from base import SavedRecommendations
from datasplit import Split


MF_SUFFIX = '_mf.pkl'
TRAIN_SUFFIX = '_train.pkl'
TEST_SUFFIX = '_rec.pkl'


def load_split(split_fname_prefix, fold=None):
    if fold is None:
        fname =  split_fname_prefix + '_split.pkl'
    else:
        fname =  split_fname_prefix + '_split_%d.pkl' % fold

    with open(fname, 'rb') as f:
        split = pkl.load(f)

    return split


def gen_mf(split, filepath, MFclass, **MFparams):
    mf = MFclass(**MFparams)
    matrices = mf.fit(split.train.get_matrix())

    fname = filepath + MF_SUFFIX
    with open(fname, 'wb') as f:
        pkl.dump(matrices, f)


def load_mf(filepath, RS):
    with open(filepath + MF_SUFFIX, 'rb') as f:
        matrices = pkl.load(f)
    RS.load_mf(matrices)
    return RS

def train_save(RS, split, out_filepath):
    RS.fit(split.train)
    RS.save(out_filepath+TRAIN_SUFFIX)


def test_save(RS, out_filepath, split):
    RS.load(out_filepath+TRAIN_SUFFIX, split.train)
    rec = SavedRecommendations()
    rec.save(out_filepath+TEST_SUFFIX, RS)


class Metrics(object):
    def __init__(self, split, filepath):
        self.RS = SavedRecommendations()
        self.RS.load(filepath+TEST_SUFFIX)
        self.split = split
        self.test_set = self.split.test
        self.metrics = dict()

    def _rlist_single_user(self, user_id, threshold):
        hidden_items = [i_id for i_id, rating in self.test_set[user_id]]
        candidate_items = list(self.split.train.get_unrated_items(user_id)) + \
                          hidden_items

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

    def list_metrics(self, threshold):
        recall = 0
        precision = 0
        F1 = 0
        n_users = len(self.test_set)
        for user_id in self.test_set:
            rlist = self._rlist_single_user(user_id, threshold)
            for atN in [1, 5, 10, 15, 20]:
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

                self.metrics['P@%d' % atN] = precision
                self.metrics['R@%d' % atN] = recall
                self.metrics['F1@%d' % atN] = F1

    def _absErr_single_rating(self, user_id, item_id, true_rating):
        pred_rating = self.RS.predict(user_id, item_id)
        absErr = np.abs(pred_rating - true_rating)
        return absErr

    def error_metrics(self):
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
        self.metrics['RMSE'] = RMSE
        self.metrics['MAE'] = MAE
        self.metrics['RMSEu'] = RMSEu
        self.metrics['MAEu'] = MAEu
