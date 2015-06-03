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
from sklearn.random_projection import GaussianRandomProjection,\
                                      SparseRandomProjection


class NeighborStrategy(object):
    __metaclass__ = abc.ABCMeta

    def _item_strategy(self, database, target_user, distances, indices,
                       zero_mean):
        similarities = np.array(1.0/(1.0 + distances)).\
            reshape(distances.shape[1])
        if np.isscalar(target_user):
            ratings = np.array([database.get_rating(target_user, item)
                                for item in indices])
        else:
            ratings = np.array([target_user[item] for item in indices])

        return (ratings, similarities)

    def _user_strategy(self, database, target_item, distances, indices,
                       zero_mean):
        similarities = np.array(1.0/(1.0 + distances)).\
            reshape(distances.shape[1])

        ratings = np.array([database.get_rating(user, target_item)
                            for user in indices])

        return (ratings, similarities)


class PredictionStrategy(object):
    __metaclass__ = abc.ABCMeta

    def _predict(self, ratings, similarities):
        if (ratings == 0).all():
            print('All user ratings on neighbor items are zero')
            pred_rating = 0
        elif np.isscalar(ratings):
            pred_rating = ratings * similarities[0]
        else:
            pred_rating = np.dot(ratings, similarities) \
                / similarities[ratings > 0].sum()
        return pred_rating


class ItemBased(RatingPredictor, NeighborStrategy, PredictionStrategy):
    def __init__(self, n_neighbors=10, algorithm='brute',
                 metric='cosine'):
        self.database = None
        self.n_neighbors = n_neighbors
        self.kNN = neighbors.kNN(n_neighbors=30*n_neighbors,
                                 algorithm=algorithm, metric=metric)

    def fit(self, database):
        self.database = database
        matrix, user_means = self.database.get_matrix(zero_mean=True)
        self.kNN.fit(matrix.T)
        return self

    def predict(self, target_user, target_item):
        item_vector = self.database.get_item_vector(target_item,
                                                    zero_mean=True)
        distances, indices = self.kNN.kneighbors(item_vector, self.n_neighbors)
        ratings, similarities = \
            self._item_strategy(self.database, target_user,
                                distances, indices, zero_mean=False)

        return self._predict(ratings, similarities)


class UserBased(RatingPredictor, PredictionStrategy):
    def __init__(self, n_neighbors=20, algorithm='brute',
                 metric='pearson'):
        self.database = None
        self.kNN = neighbors.kNN(n_neighbors=n_neighbors,
                                 algorithm=algorithm, metric=metric)

    def fit(self, database):
        self.database = database
        return self

    def predict(self, target_user, target_item):
        raise NotImplementedError('UserBased recommender not implemented')
        data, orig_indices = self.database.get_rated_users(target_item)
        self.kNN.fit(data)

        user_vector = self.database.get_user_vector(target_user)
        distances, indices = self.kNN.kneighbors(user_vector)
        indices = indices.squeeze()

        similarities = 1.0/distances
        indices = [orig_indices[i] for i in indices]

        rating = 0
        for i, user in enumerate(indices):
            rating += self.database.get_rating(user, target_item)*similarities[i]
        rating /= similarities.sum()

        return rating


class BMFrecommender(RatingPredictor, NeighborStrategy, PredictionStrategy):
    def __init__(self, neighbor_type='user', offline_kNN=True,
                 n_neighbors=10, algorithm='brute', metric='cosine',
                 threshold=0, min_coverage=1.0):
        self.database = None
        self.neighbor_type = neighbor_type
        self.threshold = threshold
        self.min_coverage = min_coverage
        self.P = None
        self.Q = None
        self.transform_matrix = None
        self.n_neighbors = n_neighbors

        if offline_kNN:
            self.model_size = 30 * n_neighbors
        else:
            self.model_size = n_neighbors

        self.kNN = neighbors.kNN(n_neighbors=self.model_size,
                                 algorithm=algorithm, metric=metric)
        self.offline_kNN = offline_kNN

    def transform(self, user_vector):
        user_vector = np.array(user_vector, ndmin=2)
        return np.dot(np.dot(user_vector, self.Q),
                      self.transform_matrix)

    def fit(self, database, P=None, Q=None):
        self.database = database
        mf = BMF(self.min_coverage)
        if P is None or Q is None:
            self.P, self.Q = \
                mf.fit(self.database.get_matrix(threshold=self.threshold))
        else:
            self.set_bmf(P, Q)

        if self.offline_kNN:
            if self.neighbor_type == 'user':
                self.kNN.fit(self.P)
            elif self.neighbor_type == 'item':
                self.kNN.fit(self.Q)
            else:
                raise ValueError('Invalid neighbor_type "%s" parameter passed \
                                 to constructor' % self.neighbor_type)
        self.transform_matrix = np.linalg.pinv(np.dot(self.Q.T, self.Q))
        return self

    def predict(self, target_user, target_item):
        if self.neighbor_type == 'user':
            if not self.offline_kNN:
                user_ids = self.database.get_rated_users(target_item)
                if len(user_ids) == 0:
                    print('No co-rating neighbor for user %d, item %d' %
                          (target_user, target_item))
                    return 0
                self.kNN.fit(self.P[user_ids, :])

            if np.isscalar(target_user):
                user_vector = self.P[target_user, :]
            else:
                user_vector = self.transform(target_user)

            if not self.offline_kNN:
                distances, indices = \
                    self.kNN.kneighbors(user_vector,
                                        min(self.n_neighbors, len(user_ids)))
            else:
                distances, indices = \
                    self.kNN.kneighbors(user_vector, self.n_neighbors)

            indices = indices.reshape(indices.shape[1])

            if not self.offline_kNN:
                indices = np.array([user_ids[i] for i in indices])

            ratings, similarities = \
                self._user_strategy(self.database, target_item,
                                    distances, indices, zero_mean=False)

        elif self.neighbor_type == 'item':
            if not self.offline_kNN:
                item_ids = self.database.get_rated_items(target_user)
                if len(item_ids) == 0:
                    print('No co-rating neighbor for user %d, item %d' %
                          (target_user, target_item))
                    return 0
                self.kNN.fit(self.Q[item_ids, :])

            item_vector = self.Q[target_item, :]

            if not self.offline_kNN:
                distances, indices = \
                    self.kNN.kneighbors(item_vector,
                                        min(self.n_neighbors, len(item_ids)))
            else:
                distances, indices = self.kNN.kneighbors(item_vector,
                                                         self.n_neighbors)

            ratings, similarities = \
                self._item_strategy(self.database, target_user,
                                    distances, indices, zero_mean=False)

        return self._predict(ratings, similarities)

    def set_bmf(self, P, Q):
        self.P = P
        self.Q = Q


class BMFRPrecommender(BMFrecommender):
    def __init__(self, RP_type='sparse', dim_red=0.5,
                 neighbor_type='user', offline_kNN=True,
                 n_neighbors=10, algorithm='brute',
                 metric='cosine', threshold=0, min_coverage=1.0):
        BMFrecommender.__init__(self, neighbor_type=neighbor_type,
                                algorithm=algorithm, metric=metric,
                                threshold=threshold, min_coverage=min_coverage)
        self.dim_red = dim_red
        self.RP = RP_type

    def transform(self, user_vector):
        new_vec = BMFrecommender.transform(self, user_vector)
        return self.RP.transform(new_vec)

    def fit(self, database, P=None, Q=None):
        self.database = database

        if self.dim_red != 'auto':
            n_components = int(np.ceil(self.dim_red*self.database.n_users()))
        else:
            n_components = self.dim_red

        if self.RP == 'gaussian':
            self.RP = GaussianRandomProjection
        elif self.RP == 'sparse':
            self.RP = SparseRandomProjection
        self.RP = self.RP(n_components=n_components)

        mf = BMF()
        if P is None or Q is None:
            self.P, self.Q = \
                mf.fit(self.database.get_matrix(threshold=self.threshold))
        else:
            self.set_bmf(P, Q)

        if self.neighbor_type == 'user':
            self.P = self.RP.fit_transform(self.P)
            if self.offline_kNN:
                self.kNN.fit(self.P)

        elif self.neighbor_type == 'item':
            self.Q = self.RP.fit_transform(self.Q)
            if self.offline_kNN:
                self.kNN.fit(self.Q)
        else:
            raise ValueError('Invalid neighbor_type parameter passed to\
                             constructor')
        self.transform_matrix = np.linalg.pinv(np.dot(self.Q.T, self.Q))
        return self
