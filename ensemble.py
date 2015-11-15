# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:21:08 2015

@author: thalita
"""
import abc
import numpy as np
from collections import Counter
from base import RatingPredictor, BaseRecommender
from scipy.stats import kendalltau
from sklearn.preprocessing.data import normalize
from sklearn.linear_model import ElasticNet
from recommender import BMFRPrecommender
from databases import HiddenRatingsDatabase
import evaluation as evalu
from utils import oneD


class BaseEnsemble(BaseRecommender):
    __metaclass__ = abc.ABCMeta

    @property
    def RS_list(self):
        "list of recommenders used by the ensemble"
        return self._RS_list

    @RS_list.setter
    def RS_list(self, val):
        self._RS_list = val

    @property
    def keep(self):
        return self._keep

    @keep.setter
    def keep(self, val):
        self._keep = val

    @property
    def diversity_measures(self):
        return self._diversity_measures

    @diversity_measures.setter
    def diversity_measures(self, val):
        self._diversity_measures = val

    @abc.abstractmethod
    def fit(self, split):
        "learn recommender model (neighborhood, matrix factorization, etc)"
        self.database = split.train
        filter_RS_list(self.RS_list, split, self.keep)
        return self

    def is_ensemble(self):
        return True

    def config(self):
        d = BaseRecommender.config(self)
        d.update(self.__dict__)
        del d['_RS_list']
        del d['_database']
        return d


def filter_RS_list(RS_list, split, keep=0.25):
        score = []
        for RS in RS_list:
            metrics = evalu.Metrics(split, RS=RS)
            metrics.def_test_set('tuning')
            metrics.error_metrics()
            score.append(metrics.metrics['RMSE_tuning'])
        score = [(s, idx) for idx, s in enumerate(score)]
        score.sort()
        keep = [idx for s, idx in score[:int(len(score)*keep)]]
        RS_list = [RS_list[idx] for idx in keep]


class RatingEnsemble(BaseEnsemble, RatingPredictor):
    __metaclass__ = abc.ABCMeta
    diversity_metric = 'stddev'

    @abc.abstractmethod
    def _rating_ensemble_strategy(self, ratings):
        pass

    def predict(self, target_user, target_item):
        ratings = [RS.predict(target_user, target_item) for RS in self.RS_list]
        self.diversity_measures += [np.std(ratings)]
        return self._rating_ensemble_strategy(ratings)

    def config(self):
        d = BaseEnsemble.config(self)
        d['diversity_metric'] = RatingEnsemble.diversity_metric
        return d

class ListEnsemble(BaseEnsemble):
    __metaclass__ = abc.ABCMeta
    diversity_metric = 'kendalltau'

    @abc.abstractmethod
    def _list_ensemble_strategy(self, rec_lists):
        pass

    def predict(self, target_user, target_item):
        return 0

    def recommend(self, target_user, how_many=np.inf, **rec_args):
        recommendations = []
        for RS in self.RS_list:
            rec_list = RS.recommend(target_user, **rec_args)
            recommendations.append(rec_list)
        recommendations = [l for l in recommendations if l != []]
        if recommendations == []:
            return []
        self.diversity_measures += [self.kendalltau(recommendations)]
        lists = self._list_ensemble_strategy(recommendations)
        how_many = min(how_many, len(lists))
        lists = lists[:how_many]
        return lists

    def kendalltau(self, lists):
        '''
        From Scipy's doc:
        Kendall’s tau is a measure of the correspondence between two rankings.
        Values close to 1 indicate strong agreement,
        values close to -1 indicate strong disagreement.
        '''
        value = 0
        count = 0
        lists = [l for l in lists if l != []]
        trim = min([len(l) for l in lists])
        for i in range(len(lists)-1):
            for j in range(1, len(lists)):
                a, b = lists[i][:trim], lists[j][:trim]
                k, p = kendalltau(a, b)
                value += k
                count += 1
        value /= count
        return value

    def config(self):
        d = BaseEnsemble.config(self)
        d['diversity_metric'] = ListEnsemble.diversity_metric
        return d


class MajorityEnsemble(ListEnsemble):
    def __init__(self, keep=1):
        self.RS_list = []
        self.diversity_measures = []
        self.keep = keep

    def _list_ensemble_strategy(self, rec_lists):
        item_votes = Counter()
        for rec_list in rec_lists:
            item_votes.update([item_id for item_id, rating in rec_list])
        return item_votes.most_common()


class RankSumEnsemble(ListEnsemble):
    def __init__(self, keep=1):
        self.RS_list = []
        self.diversity_measures = []
        self.keep = keep

    def _list_ensemble_strategy(self, rec_lists):
        rank_sum = Counter()
        for rec_list in rec_lists:
            for i, item_rating in enumerate(rec_list):
                item_id, rating = item_rating
                rank_sum[item_id] -= i
        return rank_sum.most_common()


class AvgRatingEnsemble(RatingEnsemble):
    def __init__(self, keep=1):
        self.RS_list = []
        self.diversity_measures = []
        self.keep = keep

    def _rating_ensemble_strategy(self, ratings):
        return np.mean(ratings)


class WAvgRatingEnsemble(RatingEnsemble):
    def __init__(self, keep=1):
        self.RS_list = []
        self.weights = []
        self.diversity_measures = []
        self.keep = keep

    def _rating_ensemble_strategy(self, ratings):
        ratings = np.array(ratings)
        return np.dot(self.weights, ratings)

    def fit(self, split):
        BaseEnsemble.fit(self, split)
        self.weights = []
        for RS in self.RS_list:
            metrics = evalu.Metrics(split, RS=RS)
            metrics.def_test_set('tuning')
            metrics.error_metrics()
            self.weights.append(1/metrics.metrics['RMSE_tuning'])
        self.weights = oneD(normalize(np.array(self.weights), norm='l1'))


class LinRegRatingEnsemble(RatingEnsemble):
    def __init__(self, regularization=1.0, l1_ratio=0.5, keep=1):
        self.diversity_measures = []
        self.keep = keep
        self.RS_list = []
        self.regularization = regularization
        self.l1_ratio = l1_ratio

        '''
        Elastic net performs a regularized linear regression
        It has both l1 and l2 penalties:
        alpha * [ (l1_ratio) * l1_penalty + (1-l1_ratio) * l2_penalty]
        '''
        self.model = ElasticNet(alpha=regularization,
                                l1_ratio=l1_ratio, positive=True)

    def fit(self, split):
        BaseEnsemble.fit(self, split)
        X = []
        Y = []
        for user, u in split.tuning.items():
            for item, rating in u:
                Y.append(rating)
                predictions = [RS.predict(user, item) for RS in self.RS_list]
                X.append(predictions)
        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(X, Y)

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
