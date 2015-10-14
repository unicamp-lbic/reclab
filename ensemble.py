# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:21:08 2015

@author: thalita
"""
import abc
import numpy as np
from collections import Counter
from base import BaseEnsemble, RatingPredictor
from scipy.stats import kendalltau
from sklearn.preprocessing.data import normalize
from sklearn.linear_model import ElasticNet
from recommender import BMFRPrecommender
from databases import HiddenRatingsDatabase
import evaluation as evalu
from utils import oneD


class RatingEnsemble(BaseEnsemble, RatingPredictor):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _rating_ensemble_strategy(self, ratings):
        pass

    def predict(self, target_user, target_item):
        ratings = [RS.predict(target_user, target_item) for RS in self.RS_list]
        return self._rating_ensemble_strategy(ratings)


class ListEnsemble(BaseEnsemble):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _list_ensemble_strategy(self, rec_lists):
        pass

    def predict(self, target_user, target_item):
        return 0

    def recommend(self, target_user, **rec_args):
        recommendations = []
        for RS in self.RS_list:
            rec_list = RS.recommend(target_user, **rec_args)
            recommendations.append(rec_list)
        how_many = rec_args['how_many']
        lists =  self._list_ensemble_strategy(recommendations)[:how_many]
        return lists


class MajorityEnsemble(ListEnsemble):
    def __init__(self, **varargs):
        self.RS_list = []

    def _list_ensemble_strategy(self, rec_lists):
        item_votes = Counter()
        for rec_list in rec_lists:
            item_votes.update([item_id for item_id, rating in rec_list])
        return item_votes.most_common()


class RankSumEnsemble(ListEnsemble):
    def __init__(self, **varargs):
        self.RS_list = []

    def _list_ensemble_strategy(self, rec_lists):
        rank_sum = Counter()
        for rec_list in rec_lists:
            for i, item_rating in enumerate(rec_list):
                item_id, rating = item_rating
                rank_sum[item_id] -= i
        return rank_sum.most_common()


class AvgRatingEnsemble(RatingEnsemble):
    def __init__(self, **varargs):
        self.RS_list = []
        for arg, val in varargs.items():
            self.__setattr__(arg, val)

    def _rating_ensemble_strategy(self, ratings):
        return np.mean(ratings)

    def fit(split):
        self.database = split.train

class WAvgRatingEnsemble(RatingEnsemble):
    def __init__(self, **varargs):
        self.RS_list = []
        self.weights = []

    def _rating_ensemble_strategy(self, ratings):
        ratings = np.array(ratings)
        return np.dot(self.weights, ratings)

    def fit(self, split):
        self.database = split.train
        self.weights = []
        for RS in self.RS_list:
            metrics = evalu.Metrics(split, RS=RS)
            metrics.def_test_set('valid')
            metrics.error_metrics()
            self.weights.append(1/metrics.metrics['RMSE_valid'])
        self.weights = oneD(normalize(np.array(self.weights), norm='l1'))

class LinRegRatingEnsemble(RatingEnsemble):
    def __init__(self, regularization=1.0, l1_ratio=0.5):
        self.RS_list = []
        self.regularization = regularization
        self.l1_ratio = l1_ratio
        '''
        Elastic net performs a regularized linear regression
        It has both l1 and l2 penalties:
        alpha * [ (l1_ratio) * l1_penalty + (1-l1_ratio) * l2_penalty]
        '''
        self.model = ElasticNet(alpha=regularization,
                                l1_ratio=l1_ratio)
    def fit(self, split):
        self.database = split.train
        X = []
        Y = []
        for user, u_valid in split.valid.items():
            for item, rating in u_valid:
              Y.append(rating)
              predictions = [RS.predict(user, item) for RS in self.RS_list]
              X.append(predictions)
        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(X,Y)

    def _rating_ensemble_strategy(self, ratings):
        ratings = np.array(ratings, ndmin=2)
        return float(self.model.predict(ratings))

def RPBMFEnsembleFactory(RP_type='sparse', n_projections=5,
                         dim_range=(0.25, 0.75), **BMF_args):

    dim_red = np.linspace(dim_range[0], dim_range[1], n_projections)
    RS_list = []
    for i in range(n_projections):
        RS_list.append(
            BMFRPrecommender(RP_type=RP_type, dim_red=dim_red[i],
                             **BMF_args))
    return RS_list
