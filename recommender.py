# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:48:53 2015

@author: thalita
"""

from base import RatingPredictor
import abc
import neighbors
from BMF import BMF
import numpy as np
from utils import oneD
from sklearn.random_projection import GaussianRandomProjection,\
                                      SparseRandomProjection
from sklearn.feature_extraction.text import TfidfTransformer

class NeighborStrategy(object):
    __metaclass__ = abc.ABCMeta

    def _item_strategy(self, target_user, distances, indices,
                       zero_mean):
        indices = oneD(indices)
        distances = oneD(distances)
        if np.isscalar(target_user):
            ratings = np.array([self.database.get_rating(target_user, item)
                                for item in indices])
        else:
            ratings = np.array([target_user[item]
                                for item in indices])

        if self.metric == 'cosine':
            similarities = oneD((1.0 - distances))
        elif self.metric == 'correlation':
            similarities = - distances

        return (ratings, similarities)

    def _user_strategy(self, target_item, distances, indices,
                       zero_mean):
        indices = oneD(indices)
        distances = oneD(distances)
        ratings = np.array([self.database.get_rating(user, target_item)
                            for user in indices], ndmin=1)

        if self.metric == 'cosine':
            similarities = oneD((1.0 - distances))
        elif self.metric == 'correlation':
            similarities = - distances

        return (ratings, similarities)


class PredictionStrategy(object):
    __metaclass__ = abc.ABCMeta

    def _predict(self, ratings, similarities, zero_mean=False):
        if (ratings == 0).all():
            #print('All user ratings on neighbor items are zero')
            pred_rating = 0
        else:
            ratings = oneD(ratings)
            similarities = oneD(similarities)
            denominator = similarities[ratings > 0].sum()
            if denominator != 0:
                pred_rating = np.dot(ratings, similarities)/denominator
            else:
                pred_rating = 0
        return pred_rating


class DummyRecommender(RatingPredictor):
    def __init__(self):
        self.user_means = []
        self.database = None

    def fit(self, database):
        self.database = database
        matrix, user_means = self.database.get_matrix(zero_mean=True,
                                                      sparse=True)
        self.user_means = user_means

    def predict(self, target_user, target_item):
        return self.user_means[target_user]


class ItemBased(RatingPredictor, NeighborStrategy, PredictionStrategy):
    def __init__(self, n_neighbors=10, algorithm='brute', model_size=30,
                 metric='cosine', offline_kNN=True, weighting='none',
                 **kNN_args):
        self.database = None
        self.n_neighbors = n_neighbors
        self.model_size = model_size * n_neighbors
        self.weighting = weighting
        self.metric = 'cosine'
        self.offline_kNN = offline_kNN
        self.kNN = neighbors.kNN(n_neighbors=self.model_size,
                                 algorithm=algorithm, metric=metric, **kNN_args)

    def fit(self, database):
        self.database = database
        if self.offline_kNN:
            matrix, user_means = self.database.get_matrix(zero_mean=True,
                                                          sparse=True)
            self.kNN.fit(matrix.T, keepgraph=True)
        return self

    def predict(self, target_user, target_item):
        rated_items = self.database.get_rated_items(target_user)
        n_neighbors = min(self.n_neighbors, len(rated_items))
        if len(oneD(rated_items)) == 0:
            return 0
        if self.offline_kNN:
            distances, indices = \
                self.kNN.kneighbors(target_item, n_neighbors=n_neighbors,
                                    filter=rated_items)
        else:
            matrix, user_means = self.database.get_matrix(zero_mean=True)
            item_vector = self.database.get_item_vector(target_item, zero_mean=True)
            self.kNN.fit(matrix[:, rated_items].T)
            distances, indices = \
                self.kNN.kneighbors(item_vector, n_neighbors=n_neighbors)
            indices = oneD([rated_items[i] for i in oneD(indices)])

        if len(oneD(indices)) == 0:
            return 0

        ratings, similarities = \
            self._item_strategy(target_user, distances, indices,
                                zero_mean=False)

        return self._predict(ratings, similarities)


class UserBased(RatingPredictor, NeighborStrategy, PredictionStrategy):
    def __init__(self, offline_kNN=True, weighting='none',
                 n_neighbors=20, algorithm='brute', metric='correlation',
                 **kNN_args):
        self.database = None
        self.weighting = weighting
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.offline_kNN = offline_kNN
        self.kNN = neighbors.kNN(n_neighbors=n_neighbors,
                                 algorithm=algorithm, metric=metric, **kNN_args)

    def fit(self, database):
        self.database = database
        if self.offline_kNN:
            matrix, user_means = database.get_matrix(zero_mean=True,
                                                     sparse=True)
            self.kNN.estimator.n_neighbors = matrix.shape[1]
            self.kNN.fit(matrix, keepgraph=True)
        return self

    def predict(self, target_user, target_item):
        user_indices = self.database.get_rated_users(target_item)
        if len(oneD(user_indices)) == 0:
            return 0
        if self.offline_kNN:
            distances, indices = \
                self.kNN.kneighbors(target_user, n_neighbors=self.n_neighbors,
                                filter=user_indices)
        else:
            matrix, user_means = self.database.get_matrix(zero_mean=True)
            self.kNN.fit(matrix[user_indices, :])
            user_vector = self.database.get_user_vector(target_user, zero_mean=True)
            n_neighbors = min(self.n_neighbors, len(user_indices))
            distances, indices = \
                self.kNN.kneighbors(user_vector, n_neighbors)
            indices = np.array([user_indices[i] for i in oneD(indices)])

        if len(oneD(indices)) == 0:
            return 0

        ratings, similarities = \
            self._user_strategy(target_item, distances, indices,
                                zero_mean=False)

        return self._predict(ratings, similarities)


class MFrecomender(RatingPredictor):
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def __MF_args__(RS_args):
        pass

    def load_mf(self, P, Q):
        self.P = P
        self.Q = Q

    @abc.abstractclassmethod
    def gen_mf(self):
        pass


class BMFrecommender(MFrecomender, NeighborStrategy, PredictionStrategy):
    __MF_type__ = BMF

    def __MF_args__(RS_args):
        args = ['min_coverage', 'bin_threshold']
        return dict([(arg, RS_args[arg]) for arg in args])

    def __init__(self, neighbor_type='user', offline_kNN=False,
                 bin_threshold=0, min_coverage=1.0, weighting='none',
                 n_neighbors=10, algorithm='brute', metric='cosine',
                 model_size=30, **kNN_args):
        self.database = None
        self.weighting = weighting
        self.metric = metric
        self.neighbor_type = neighbor_type
        self.bin_threshold = bin_threshold
        self.min_coverage = min_coverage
        self.P = None
        self.Q = None
        self.n_neighbors = n_neighbors

        if offline_kNN:
            self.model_size = model_size * n_neighbors
        else:
            self.model_size = n_neighbors

        self.kNN = \
            neighbors.kNN(n_neighbors=self.model_size,
                          algorithm=algorithm, metric=metric, **kNN_args)
        self.offline_kNN = offline_kNN
        self.kNN_graph = None

    def transform(self, user_vector):
        raise NotImplementedError()
        items = set(np.where(oneD(user_vector) > self.bin_threshold)[0])
        orig_len = len(items)
        factors = [(set(np.where(line == 1)[0]), i)
                   for i, line in enumerate(self.Q.T)]
        user_factors = []
        for f, i in factors:
            if f.issubset(items):
                user_factors.append(i)
                items.difference_update(f)
        user_vector = [1 if i in set(user_factors) else 0
                       for i in range(self.P.shape[1])]
        if len(items)/orig_len > (1 - self.min_coverage):
            factors = [(len(f.intersection(items))/len(f.union(items)), f, i)
                       for f, i in factors]
            factors.sort(reverse=True)
            for jac, factor, idx in factors:
                if len(items) == 0:
                    break
                new = items.difference(factor)
                if len(new) < len(items):
                    items = new
                    user_vector[idx] = jac

        return np.array(user_vector)

    def gen_mf(self, database):
        self.database = database
        mf = BMF(self.min_coverage)
        self.P, self.Q = \
                mf.fit(self.database.get_matrix(threshold=self.bin_threshold))
        return self.P, self.Q

    def fit(self, database):
        self.database = database
        if self.P is None or  self.Q is None:
            self.gen_mf(self.database)

        if self.weighting == 'tf-idf':
            tfidf = TfidfTransformer(norm=None, use_idf=True,
                                     smooth_idf=True, sublinear_tf=False)
            self.P = tfidf.fit_transform(self.P)
            self.Q = tfidf.fit_transform(self.Q)

        if self.offline_kNN:
            if self.neighbor_type == 'user':
                self.kNN.fit(self.P, keepgraph=True)
            elif self.neighbor_type == 'item':
                self.kNN.fit(self.Q, keepgraph=True)
            else:
                raise ValueError('Invalid neighbor_type "%s" parameter passed \
                                 to constructor' % self.neighbor_type)
        return self

    def predict(self, target_user, target_item):
        if self.neighbor_type == 'user':
            if np.isscalar(target_user):
                user_vector = self.P[target_user, :]
            else:
                user_vector = self.transform(target_user)

            user_ids = self.database.get_rated_users(target_item)
            if len(user_ids) == 0:
                #print('No co-rating neighbor for user %d, item %d' %
                #      (target_user, target_item))
                return 0

            if self.offline_kNN:
                n_neighbors = min(self.n_neighbors, self.database.n_users())
                distances, indices = \
                    self.kNN.kneighbors(target_user, n_neighbors,
                                        filter=user_ids)
            else:
                self.kNN.fit(self.P[user_ids, :])
                n_neighbors = min(self.n_neighbors, len(user_ids))
                distances, indices = \
                    self.kNN.kneighbors(user_vector, n_neighbors)
                indices = np.array([user_ids[i] for i in oneD(indices)])

            ratings, similarities = \
                self._user_strategy(target_item, distances, indices,
                                    zero_mean=False)

        elif self.neighbor_type == 'item':
            item_vector = self.Q[target_item, :]
            item_ids = self.database.get_rated_items(target_user)
            if len(item_ids) == 0:
                #print('No co-rating neighbor for user %d, item %d' %
                #      (target_user, target_item))
                return 0

            if self.offline_kNN:
                n_neighbors = min(self.n_neighbors, self.database.n_items())
                distances, indices = \
                    self.kNN.kneighbors(item_vector, n_neighbors,
                                        filter=item_ids)
            else:
                self.kNN.fit(self.Q[item_ids, :])
                n_neighbors = min(self.n_neighbors, len(item_ids))
                distances, indices = \
                    self.kNN.kneighbors(item_vector, n_neighbors)
                indices = np.array([item_ids[i] for i in oneD(indices)])

            ratings, similarities = \
                self._item_strategy(target_user,
                                    distances, indices, zero_mean=False)

        return self._predict(ratings, similarities)


class BMFRPrecommender(BMFrecommender):
    def __init__(self, RP_type='sparse', dim_red=0.5,
                 **BMF_args):
        BMFrecommender.__init__(self, **BMF_args)
        self.dim_red = dim_red
        self.RP = RP_type

    def transform(self, user_vector):
        new_vec = BMFrecommender.transform(self, user_vector)
        return self.RP.transform(new_vec)

    def fit(self, database):
        if self.P is None or self.Q is None:
            mf = BMF(self.min_coverage)
            self.P, self.Q = \
                mf.fit(self.database.get_matrix(threshold=self.bin_threshold))

        if self.weighting == 'tf-idf':
            tfidf = TfidfTransformer(norm=None, use_idf=True,
                                     smooth_idf=True, sublinear_tf=False)
            self.P = tfidf.fit_transform(self.P)
            self.Q = tfidf.fit_transform(self.Q)

        if self.dim_red != 'auto':
            n_components = int(np.ceil(self.dim_red*self.P.shape[1]))
        else:
            n_components = self.dim_red

        if self.RP == 'gaussian':
            self.RP = GaussianRandomProjection(n_components=n_components)
        elif self.RP == 'sparse':
            self.RP = SparseRandomProjection(n_components=n_components)

        if self.neighbor_type == 'user':
            self.P = self.RP.fit_transform(self.P)
            if self.offline_kNN:
                self.kNN.fit(self.P, keepgraph=True)

        elif self.neighbor_type == 'item':
            self.Q = self.RP.fit_transform(self.Q)
            if self.offline_kNN:
                self.kNN.fit(self.Q, keepgraph=True)
        else:
            raise ValueError('Invalid neighbor_type parameter passed to\
                             constructor')
        return self
