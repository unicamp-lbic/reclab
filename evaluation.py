# -*- coding: utf-8 -*-
"""
Created on Tue May  5 18:29:23 2015

@author: thalita
"""
import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
import os
from pickle import dump, load
from numpy.random import choice
from databases import SubDatabase, HiddenRatingsDatabase


class Evaluator(object):
    def __init__(self, RS, test_set, pct_hidden=0.2, topk=5, threshold=0):
        self.RS = RS
        self.test_set = test_set
        self.pct_hidden = pct_hidden
        self.topk = topk
        self.threshold = threshold

    def _split_positive_negative(self, user_vector):
        '''
        Split items in positive and negative evaluations
        To be used in P&R calculation
        '''
        u_positives = []
        u_negatives = []
        for i, rating in enumerate(user_vector):
            if rating >= self.threshold:
                u_positives.append(i)
            elif rating > 0:
                u_negatives.append(i)
        return (u_positives, u_negatives)

    @staticmethod
    def columns():
        return ['P','R','F1','MAE','RMSE']


    def avg_all_users(self):
        result = [self.single_user(user) for user in self.test_set]
        result = sum(result)
        result /= len(self.test_set)
        return result


    def single_user(self, user_vector):
        u_positives, u_negatives = self._split_positive_negative(user_vector)

        # Hide some items to check on them later
        random_pick = lambda aList: list(
                   choice(aList,
                   np.ceil(self.pct_hidden*len(aList)),
                   replace=False)) if aList != [] else aList
        hidden_positives = random_pick(u_positives)  # u and Ihid
        hidden = hidden_positives + random_pick(u_negatives)

        new_vector = [0 if i in hidden else rating
                      for i, rating in enumerate(user_vector)]
        unrated = [i for i, rating in enumerate(new_vector) if rating == 0]

        #generate recomendations
        rlist = self.RS.recommend(new_vector, how_many=len(unrated),
                                  threshold=self.threshold,
                                  candidate_items=unrated)
        predictions = dict(rlist)

        # Calculate precision and recall
        # r and Ihid
        pred_hidden = set(hidden) & set(rlist)
        # r and u and Ihid
        pred_hidden_positives = pred_hidden & set(hidden_positives)
        # Recall:
        if len(hidden_positives) > 0:
            recall = len(pred_hidden_positives)/float(len(hidden_positives))
        else:
            recall = 1.0
        # Precision:
        if len(pred_hidden) > 0:
            precision = len(pred_hidden_positives)/float(len(pred_hidden))
        elif len(u_positives) == 0:
            precision = 1.0
        else:
            precision = 0.0
        if precision+recall > 0:
            F1 = precision*recall/(precision+recall)
        else:
            F1 = 0.0

        # Calculate MAE and RMSE
        deviations = np.array([user_vector[i]-predictions[i]
                               for i in pred_hidden])
        MAE = np.mean(deviations)
        RMSE = np.sqrt(np.mean(deviations**2))

        metrics = np.array([precision, recall, F1, MAE, RMSE])
        return metrics

class HoldoutRatingsView(object):
    def __init__(self, database, testset_folder, pct_hidden=0.2, threshold=4):
        self.pct_hidden = pct_hidden
        self.threshold = threshold
        self.folder = testset_folder
        HOLDOUT_FILE = testset_folder + \
            '/%d_pct_hidden' % (100 * pct_hidden) + \
            '_ratings_%d+.pkl' % (threshold)

        if os.path.isfile(HOLDOUT_FILE):
            with open(HOLDOUT_FILE, 'rb') as f:
                self.hidden_coord = load(f)
        else:
            # get positions equal to or above threshold (ratings)
            matrix = np.array(database.get_matrix())
            row, col = np.where(matrix >= threshold)
            n_hidden = np.ceil(pct_hidden*len(row))
            # pick n_hidden random positions
            hidden_idx = np.random.randint(0, len(row), n_hidden)
            self.hidden_coord = [coo for coo in zip(row[hidden_idx],
                                               col[hidden_idx])]
            with open(HOLDOUT_FILE, 'wb') as f:
                dump(self.hidden_coord, f)

        self.train_set = HiddenRatingsDatabase(database.get_matrix(),
                                                self.hidden_coord)
        self.test_set =  [(database.get_matrix()[u, i], u, i)
                          for u, i in self.hidden_coord]


class HoldoutRatingsMetrics(object):
    def __init__(self, RS, test_set, topk, threshold):
        self.test_set = test_set
        self.RS = RS
        self.topk = topk
        self.threshold = threshold
        self.test_set = test_set

    def _metrics_single_rating(self, user_id, item_id, true_rating):
        unrated = self.RS.database.get_unrated_items(user_id)
        qtty_candidates = min(np.ceil(0.1*self.RS.database.n_items()), 1000)
        candidate_items = choice(unrated, size=qtty_candidates,
                                 replace=False).tolist() + [item_id]

        rlist = self.RS.recommend(user_id, how_many=self.topk,
                                  threshold=self.threshold,
                                  candidate_items=candidate_items)

        rlist = dict(rlist)
        hit = 1 if item_id in rlist else 0

        pred_rating = self.RS.predict(user_id, item_id)
        absErr = np.abs(pred_rating - true_rating)

        return hit, absErr

    @staticmethod
    def columns():
        return ['P','R','F1','MAE','RMSE']

    def calc_metrics(self):
        total_hits = 0
        sum_absErr = 0
        sum_sqErr = 0
        for rating, user, item in self.test_set:
            hit, absErr = self._metrics_single_rating(rating, user, item)
            total_hits += hit
            sum_absErr += absErr
            sum_sqErr += absErr**2
        # According to Cremonesi et al. 2010
        # Recall = #hits/|test set|
        # Precision = #hits/(topk * |test set|) = recall/topk
        recall = total_hits/len(self.test_set)
        precision = recall/self.topk
        if precision+recall > 0:
            F1 = precision*recall/(precision+recall)
        else:
            F1 = 0.0
        MAE = sum_absErr/len(self.test_set)
        RMSE = np.sqrt(sum_sqErr)/len(self.test_set)

        metrics = np.array([precision, recall, F1, MAE, RMSE])
        return metrics

class HoldoutRatingsEvaluator(object):
    def __init__(self, holdout_view, RS_type, RS_arguments, result_folder,
                 topk=1, threshold=3):
        self.RS = RS_type(**RS_arguments)

        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
        self.pct_hidden = holdout_view.pct_hidden
        self.topk = topk
        self.threshold = threshold
        self.fname_prefix = result_folder+'/'+\
                            '_'.join([RS_type.__name__]+
                                     [str(arg) for arg in RS_arguments]) \
                            + '_%d_pct_hidden' % (100 * self.pct_hidden) \
                            + '_rec_threshold_%d' % self.threshold \
                            + '_top_%d' % self.topk
        self.holdout = holdout_view


    def train(self, force_train=False):
        train_file = self.fname_prefix + '_trained.pkl'
        if os.path.isfile(train_file) and not force_train:
            with open(train_file, 'rb') as f:
                self.RS = load(f)
        else:
            self.RS.fit(self.holdout.train_set)
            with open(train_file, 'wb') as f:
                dump(self.RS, f)

    def test(self):
        evalu = HoldoutRatingsMetrics(self.RS, self.holdout.test_set, self.topk,
                                      self.threshold)
        metrics = evalu.calc_metrics()
        metrics_labels = evalu.columns()
        np.savetxt(self.fname_prefix+'_test.txt', metrics.T, delimiter=',',
                   header=','.join(['"'+l+'"' for l in metrics_labels]))

        return metrics

class HoldoutBMF(HoldoutRatingsEvaluator):
    def __init__(self, holdout_view, RS_type, RS_arguments, result_folder,
                 topk=1, threshold=3):
        HoldoutRatingsEvaluator.__init__(self, holdout_view, RS_type, RS_arguments,
                                         result_folder, topk, threshold)

        min_coverage = RS_arguments['min_coverage']
        self.BMF_file = self.holdout.folder + \
            'BMF_coverage_%0.2f' % min_coverage + \
            '_%d_pct_hidden.pkl' % (holdout_view.pct_hidden * 100)

    def train(self, force_train=False):
        train_file = self.fname_prefix + '_trained.pkl'
        BMF_file = self.BMF_file
        if os.path.isfile(train_file) and not force_train:
            with open(train_file, 'rb') as f:
                self.RS = load(f)
        else:
            if os.path.isfile(BMF_file) and not force_train:
                with open(BMF_file, 'rb') as f:
                    P, Q = load(f)
                self.RS.fit(self.holdout.train_set, P, Q)

            else:
                self.RS.fit(self.holdout.train_set)
                with open(BMF_file, 'wb') as f:
                    dump((self.RS.P, self.RS.Q), f)
                with open(train_file, 'wb') as f:
                    dump(self.RS, f)


class kFoldView(object):
    def __init__(self, database, kfold_folder, n_folds=5, shuffle=True):

        self.kfold = self._gen_kfold(kfold_folder, database.n_users(),
                                     n_folds=n_folds, shuffle=shuffle)
        self.kfold_folder = kfold_folder

        self.n_folds = n_folds

        self.train_sets = [SubDatabase(database.matrix, self.kfold[i][0])
                           for i in range(n_folds)]
        self.test_sets = [SubDatabase(database.matrix, self.kfold[i][1])
                          for i in range(n_folds)]

    def _gen_kfold(self, folder, size, n_folds, shuffle=True):
        KFOLD_FILE = folder + '/%d_fold_indices.pkl' % n_folds
        if os.path.isfile(KFOLD_FILE):
            with open(KFOLD_FILE, 'rb') as f:
                kfold_gen = load(f)
        else:
            kfold_gen = list(KFold(size, n_folds=n_folds, shuffle=shuffle))
            with open(KFOLD_FILE, 'wb') as f:
                dump(kfold_gen, f)
        return kfold_gen

    def unpack(self):
        return (self.kfold, self.n_folds, self.train_sets, self.test_sets)


class kFoldEvaluator(object):
    def __init__(self, kfold_view, RS_type, RS_arguments, result_folder,
                 pct_hidden=0.2, topk=10, threshold=0):
        self.kfold, self.n_folds, self.train_sets, self.test_sets = \
            kfold_view.unpack()
        self.kfold_view = kfold_view
        self.RS = [RS_type(**RS_arguments) for _ in range(self.n_folds)]

        if not os.path.isdir(result_folder):
            os.mkdir(result_folder)
        self.metrics_labels = []
        self.pct_hidden = pct_hidden
        self.topk = topk
        self.threshold = threshold
        self.fname_prefix = result_folder+'/'+\
                            '_'.join([RS_type.__name__]+
                                     [str(arg) for arg in RS_arguments]) \
                            + '_%dfolds' % self.n_folds \
                            + '_%0.2f_pct_hidden' % self.pct_hidden \
                            + 'top_%d_threshold_%d' % (self.topk, self.threshold)

    def _train_single_fold(self, i, force_train):
        train_file = self.fname_prefix + '_trained_fold_%d.pkl'%i
        if os.path.isfile(train_file) and not force_train:
            with open(train_file, 'rb') as f:
                self.RS[i] = load(f)
        else:
            self.RS[i].fit(self.train_sets[i])
            with open(train_file, 'wb') as f:
                dump(self.RS[i], f)

    def train(self, force_train=False):
        for  i in range(len(self.RS)):
            self._train_single_fold(i, force_train)

    def _test_single_fold(self, i):
        evalu = Evaluator(self.RS[i], self.test_sets[i].get_matrix(),
                          self.pct_hidden, self.topk,
                          self.threshold)
        metrics = evalu.avg_all_users()
        self.metrics_labels = evalu.columns()
        return metrics

    def test(self):
        metrics = []
        for i in range(len(self.RS)):
            metrics.append(self._test_single_fold(i))
        metrics = np.array(metrics)
        metrics = pd.DataFrame(metrics, columns=self.metrics_labels)
        metrics.to_csv(self.fname_prefix+'_test.txt')
        return metrics


class kFoldBMF(kFoldEvaluator):
    def __init__(self, kfold_view, RS_type, RS_arguments, result_folder,
                 pct_hidden=0.2, topk=10, threshold=0):

        kFoldEvaluator.__init__(self, kfold_view, RS_type, RS_arguments,
                                result_folder, pct_hidden, topk, threshold)

        min_coverage = RS_arguments['min_coverage']
        self.BMF_file_prefix = self.kfold_view.kfold_folder + \
            'BMF_coverage_%0.2f' % min_coverage


    def _train_single_fold(self, i, force_train):
        train_file = self.fname_prefix + '_trained_fold_%d.pkl' % i
        BMF_file = self.BMF_file_prefix + '_fold_%d' % i

        if os.path.isfile(train_file) and not force_train:
            with open(train_file, 'rb') as f:
                self.RS[i] = load(f)
        else:
            if os.path.isfile(BMF_file) and not force_train:
                with open(BMF_file, 'rb') as f:
                    P, Q = load(f)
                self.RS[i].fit(self.train_sets[i], P, Q)
            else:
                self.RS[i].fit(self.train_sets[i])
                with open(BMF_file, 'wb') as f:
                    dump((self.RS[i].P, self.RS[i].Q), f)
                with open(train_file, 'wb') as f:
                    dump(self.RS[i], f)


