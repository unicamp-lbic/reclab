# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:21:08 2015

@author: thalita
"""
import abc
from collections import Counter
from base import BaseEnsemble, RatingPredictor
from scipy.stats import kendalltau
from recommender import BMFRPrecommender
import numpy as np


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
        RS_factory = varargs['RS_factory']
        RS_args = varargs
        del RS_args['RS_factory']
        self.RS_list = RS_factory(**RS_args)

    def _list_ensemble_strategy(self, rec_lists):
        item_votes = Counter()
        for rec_list in rec_lists:
            item_votes.update([item_id for item_id, rating in rec_list])
        return item_votes.most_common()


class RankSumEnsemble(ListEnsemble):
    def __init__(self, **varargs):
        RS_factory = varargs['RS_factory']
        RS_args = varargs
        del RS_args['RS_factory']
        self.RS_list = RS_factory(**RS_args)

    def _list_ensemble_strategy(self, rec_lists):
        rank_sum = Counter()
        for rec_list in rec_lists:
            for i, item_rating in enumerate(rec_list):
                item_id, rating = item_rating
                rank_sum[item_id] -= i
        return rank_sum.most_common()


class AvgRatingEnsemble(RatingEnsemble):
    def __init__(self, **varargs):
        RS_factory = varargs['RS_factory']
        RS_args = varargs
        del RS_args['RS_factory']
        self.RS_list = RS_factory(**RS_args)

    def _rating_ensemble_strategy(self, ratings):
        return np.mean(ratings)


def RPBMFEnsembleFactory(RP_type='sparse', n_projections=5,
                         dim_range=(0.25, 0.75), **BMF_args):

    dim_red = np.linspace(dim_range[0], dim_range[1], n_projections)
    RS_list = []
    for i in range(n_projections):
        RS_list.append(
            BMFRPrecommender(RP_type=RP_type, dim_red=dim_red[i],
                             **BMF_args))
    return RS_list
