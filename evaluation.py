# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 17:58:04 2015

@author: thalita
"""
import numpy as np
from base import SavedRecommendations
from collections import defaultdict
from utils import to_gzpickle, read_gzpickle


MF_SUFFIX = '_mf.pkl'
TRAIN_SUFFIX = '_train.pkl'
REC_SUFFIX = '_rec.pkl'

FINAL_MF_SUFFIX = '_mf_final.pkl'
FINAL_TRAIN_SUFFIX = '_train_final.pkl'
FINAL_REC_SUFFIX = '_rec_final.pkl'

def merge_train_valid(split):
    for user in split.valid:
        for item, rating in split.valid[user]:
            split.train.set_rating(user, item, rating)

def load_split(split_fname_prefix, fold=None):
    if fold is None:
        fname = split_fname_prefix + '_split.pkl'
    else:
        fname = split_fname_prefix + '_split_%d.pkl' % fold

    split = read_gzpickle(fname)

    return split


def gen_mf(split, filepath, RS, final=False, **MF_args):
    if final:
        merge_train_valid(split)
    matrices = RS.gen_mf(split.train, **MF_args)
    fname = filepath + (MF_SUFFIX if not final else FINAL_MF_SUFFIX)
    to_gzpickle(matrices, fname)


def load_mf(filepath, RS, final=False):
    fname = filepath + (MF_SUFFIX if not final else FINAL_MF_SUFFIX)
    matrices = read_gzpickle(fname)
    RS.load_mf(*matrices)
    return RS


def train_save(RS, split, out_filepath, final=False):
    if final:
        merge_train_valid(split)
    RS.fit(split.train)
    if not final:
        out_name = out_filepath+TRAIN_SUFFIX
    else:
        out_name = out_filepath+FINAL_TRAIN_SUFFIX
    RS.save(out_name)


def rec_save(RS, out_filepath, split, final=False):
    if final:
        for user in split.valid:
            for item, rating in split.valid[user]:
                split.train.set_rating(user, item, rating)
    if not final:
        RS.load(out_filepath+TRAIN_SUFFIX, split.train)
        out_name = out_filepath+REC_SUFFIX
    else:
        RS.load(out_filepath+FINAL_TRAIN_SUFFIX, split.train)
        out_name = out_filepath+FINAL_REC_SUFFIX
    rec = SavedRecommendations()
    rec.save(out_name, RS)


def ensemble_train_save(ens, out_filepath, split, final=False):
    if final:
        for user in split.valid:
            for item, rating in split.valid[user]:
                split.train.set_rating(user, item, rating)
    ens.fit(split)
    if not final:
        out_name = out_filepath+TRAIN_SUFFIX
    else:
        out_name = out_filepath+FINAL_TRAIN_SUFFIX
    ens.save(out_name)



def ensemble_rec_save(ens, out_filepath, split, final=False):
    if final:
        for user in split.valid:
            for item, rating in split.valid[user]:
                split.train.set_rating(user, item, rating)
    rec = SavedRecommendations()
    if not final:
        ens.load(out_filepath+TRAIN_SUFFIX, split.train)
        out_name = out_filepath+REC_SUFFIX
    else:
        ens.load(out_filepath+FINAL_TRAIN_SUFFIX, split.train)
        out_name = out_filepath+FINAL_REC_SUFFIX
    rec.save(out_name, ens)


def load_model(RS, out_filepath, split, final=False):
    if not final:
        out_name = out_filepath+TRAIN_SUFFIX
    else:
        out_name = out_filepath+FINAL_TRAIN_SUFFIX
    RS.load(out_name, split.train)


def load_recommendations(filepath, final=False):
    if not final:
        out_name = filepath+REC_SUFFIX
    else:
        out_name = filepath+FINAL_REC_SUFFIX
    rec = SavedRecommendations()
    rec.load(out_name)
    return rec


class Metrics(object):
    __atN__ = [1, 5, 10, 15, 20, 30, 50]

    def ir_metric_names(which, atNs=None):
        if atNs is None:
            atNs = Metrics.__atN__
        metrics = ['P@%d_' % atN + which for atN in atNs]
        metrics += ['R@%d_' % atN + which for atN in atNs]
        metrics += ['F1@%d_' % atN + which for atN in atNs]
        return metrics

    def error_metric_names(which, user=False):
        return ['RMSE' + ('u' if user else '') + '_' + which,
                'MAE' + ('u' if user else '') + '_' + which]

    def coverage_metric_names(which):
        return ['user_coverage',
                'item_coverage']

    def ensemble_metrics_names():
        return ['kendalltau', 'stddev']

    def __init__(self, split, filepath=None, RS=None, final=False):
        self.RS = SavedRecommendations()
        if RS is not None:
            self.RS = RS
        elif filepath is not None:
            if not final:
                self.RS.load(filepath+REC_SUFFIX)
            else:
                self.RS.load(filepath+FINAL_REC_SUFFIX)
        else:
            raise ValueError('Must inform either path to recommender\
            or a recommender object')
        self.split = split
        if final:
            merge_train_valid(self.split)
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
        elif which == 'tuning':
            self.test_set = self.split.tuning
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
        # and coverage too
        if self.test_set is None:
            raise ValueError('def_test_set must be called before metrics \
            computation')
        MAE = 0
        MSE = 0
        MAEu = 0
        MSEu = 0
        MSEr = defaultdict(list)
        nUsers = len(self.test_set)
        nTestRatings = sum([len(r) for r in self.test_set.values()])
        for user, test in self.test_set.items():
            for item, rating in test:
                absErr = self._absErr_single_rating(user, item, rating)
                MSEr[rating].append(absErr)
                MAE += absErr/nTestRatings
                MSE += absErr**2/nTestRatings
                MAEu += absErr/len(test)
                MSEu += absErr**2/len(test)
        MAEu /= nUsers
        MSEu /= nUsers
        RMSE = np.sqrt(MSE)
        RMSEu = np.sqrt(MSEu)
        for r in MSEr:
            self.metrics['RMSEr%d_' % r + self.which] \
                = np.sqrt(np.mean(MSEr[r]))

        self.metrics['RMSE_' + self.which] = RMSE
        self.metrics['MAE_' + self.which] = MAE
        self.metrics['RMSEu_' + self.which] = RMSEu
        self.metrics['MAEu_' + self.which] = MAEu

    def coverage_metrics(self):
        nUsers = self.RS.config['n_users']
        nItems = self.RS.config['n_items']
        covered_items = (self.RS.pred_ratings > 0).any(axis=0).sum()
        covered_users = (self.RS.pred_ratings > 0).any(axis=1).sum()
        # User coverage
        self.metrics['user_coverage'] = covered_users/nUsers
        # Catalog (item) coverage
        self.metrics['item_coverage'] = covered_items/nItems


    def ensemble_metrics(self):
        self.metrics[self.RS.config['diversity_metric']] =\
            np.mean(self.RS.config['_diversity_measures'])
