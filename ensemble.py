# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:21:08 2015

@author: thalita
"""
import abc
from collections import Counter
from base import BaseEnsemble
from Recommender import BMFRPrecommender
import numpy as np


class RatingEnsemble(BaseEnsemble):
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
    def _list_ensemble_strategy(rec_lists):
        pass

    def recommend(self, target_user, how_many, threshold):
        recommendations = []
        for RS in self.RS_list:
            rec_list = RS.recommend(target_user, how_many, threshold)
            recommendations.append(rec_list)
        return self._list_ensemble_strategy(recommendations)[:how_many]


class MajorityEnsemble(ListEnsemble):
    def __init__(self, database, RS_list):
        self.RS_list = RS_list
        self.database = database

    def _list_ensemble_strategy(self, rec_lists):
        item_votes = Counter()
        for rec_list in rec_lists:
            item_votes.update([item_id for item_id, rating in rec_list])
        return item_votes.most_common()


class RankSumEnsemble(ListEnsemble):
    def __init__(self, database, RS_list):
        self.RS_list = RS_list
        self.database = database

    def _list_ensemble_strategy(self, rec_lists):
        rank_sum = Counter()
        for rec_list in rec_lists:
            for i, item_id in enumerate(rec_list):
                rank_sum[item_id] -= i
        return rank_sum.most_common()


class AvgRatingEnsemble(RatingEnsemble):
    def __init__(self, database, RS_list):
        self.RS_list = RS_list
        self.database = database

    def _rating_ensemble_strategy(ratings):
        return np.mean(ratings)


class RPBMFEnsembleFactory(BaseEnsemble):
    def __init__(self, BMF_recommender, ensemble_type, n_projections=5,
                 dim_range=(0.3, 0.5)):
        dim = np.linspace(dim_range[0],
                        dim_range[1],
                        n_projections)
        self.RS_list = []
        for i in range(n_projections):
            self.RS_list.append(BMFRPrecommender(BMF_recommender=BMF_recommender,
                                            RP_type='sparse',
                                            dim_red=dim[i]))
        self.ensemble_type = ensemble_type
        self.database = BMF_recommender.database

    def run(self):
       return self.ensemble_type(database=self.database,
                             RS_list=self.RS_list)
