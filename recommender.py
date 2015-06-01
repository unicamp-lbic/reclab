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

class ItemStrategy(object):
    __metaclass__ = abc.ABCMeta

    @staticmethod
    def _item_strategy(database, target_user, distances, indices, zero_mean):
        indices = indices.squeeze()
        similarities = np.array(1.0/(1.0 + distances)).squeeze()
        if np.isscalar(target_user):
            ratings = np.array([database.get_rating(target_user, item)
                                for item in indices])
        else:
            ratings = np.array([target_user[item] for item in indices])

        if (ratings == 0).all():
            print('All user ratings on neighbor items are zero')
            pred_rating = 0
        else:
            pred_rating = np.dot(ratings, similarities) \
                / similarities[ratings > 0].sum()
        return pred_rating

class ItemBased(RatingPredictor, ItemStrategy):
    def __init__(self, n_neighbors=30, algorithm='brute',
                 metric='cosine'):
        self.database = None
        self.kNN = neighbors.kNN(n_neighbors=n_neighbors,
                                 algorithm=algorithm, metric=metric)

    def fit(self, database):
        self.database = database
        matrix, user_means = self.database.get_matrix(zero_mean=True)
        self.kNN.fit(matrix.T)
        return self

    def predict(self, target_user, target_item):
        item_vector = self.database.get_item_vector(target_item,
                                                    zero_mean=True)
        distances, indices = self.kNN.kneighbors(item_vector)
        return self._item_strategy(self.database, target_user,
                            distances, indices, zero_mean=False)



class UserBased(RatingPredictor):
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


class BMFrecommender(RatingPredictor, ItemStrategy):
    def __init__(self, neighbor_type='user', offline_kNN=True,
                 n_neighbors=20, algorithm='brute', metric='cosine',
                 threshold=0, min_coverage=1.0):
        self.database = None
        self.neighbor_type = neighbor_type
        self.threshold = threshold
        self.min_coverage = min_coverage
        self.P = None
        self.Q = None
        self.transform_matrix = None
        self.kNN = neighbors.kNN(n_neighbors=n_neighbors,
            algorithm=algorithm, metric=metric)
        self.offline_kNN = offline_kNN

    def transform(self, user_vector):
        np.user_vector.shape = (1,len(user_vector))
        return np.dot(self.transform_matrix, user_vector)


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
                raise ValueError('Invalid neighbor_type parameter passed to\
                                 constructor')
        self.transform_matrix = np.linalg.pinv(np.dot(self.Q.T, self.Q))
        return self

    def predict(self, target_user, target_item):
        if self.neighbor_type == 'user':
            if not self.offline_kNN:
                user_ids = self.database.get_rated_users(target_item)
                self.kNN.fit(self.P[user_ids, :])

            if np.isscalar(target_user):
                user_vector = self.P[target_user, :]
            else:
                user_vector = self.transform(target_user)

            distances, indices = self.kNN.kneighbors(user_vector)
            indices = indices.squeeze()

            if not self.offline_kNN:
                indices = [user_ids[i] for i in indices]

            ratings = [self.database.get_rating(user, target_item)
                   for user in indices]

        elif self.neighbor_type == 'item':
            if not self.offline_kNN:
                item_ids = self.dataset.get_rated_items(target_user)
                self.kNN.fit(self.Q[item_ids, :])

            item_vector = self.Q[target_item, :]
            distances, indices = self.kNN.kneighbors(item_vector)

            pred_rating = \
            self._item_strategy(self.database, target_user,
                                distances, indices, zero_mean=False)

        return pred_rating

    def set_bmf(self, P, Q):
        self.P = P
        self.Q = Q


class BMFRPrecommender(BMFrecommender):
    def __init__(self, RP_type='sparse', dim_red='auto',
                 neighbor_type='user', offline_kNN=True,
                 n_neighbors=10, algorithm='brute',
                 metric='cosine', threshold=0):
        BMFrecommender.__init__(self, n_neighbors, algorithm,
                                metric, threshold)
        if dim_red != 'auto':
            n_components = int(np.ceil(dim_red*self.database.n_users()))
        else:
            n_components = dim_red

        if RP_type == 'gaussian':
            RP_type = GaussianRandomProjection
        elif RP_type == 'sparse':
            RP_type = SparseRandomProjection
        self.RP = RP_type(n_components=n_components)

    def fit(self, database, P=None, Q=None):
        self.database = database
        mf = BMF()
        if P is None or Q is None:
            self.P, self.Q = \
                mf.fit(self.database.get_matrix(threshold=self.threshold))
        else:
            self.set_bmf(P, Q)

        if self.neighbor_type == 'user':
            self.P = self.RP.fit_transform(self.P)
            if self.offline_knn:
                self.kNN.fit(self.P)

        elif self.neighbor_type == 'item':
            self.Q = self.RP.fit_transform(self.Q)
            if self.offline_knn:
                self.kNN.fit(self.Q)
        else:
            raise ValueError('Invalid neighbor_type parameter passed to\
                             constructor')
        self.transform_matrix = np.linalg.pinv(np.dot(self.Q.T, self.Q))
        return self
