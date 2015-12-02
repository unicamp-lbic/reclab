# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 15:48:53 2015

@author: thalita
"""

from base import RatingPredictor
import abc
import neighbors
from BMF import BMF
from mf import SVD
import numpy as np
from utils import oneD, RANDOM_SEED
from sklearn.random_projection import GaussianRandomProjection,\
                                      SparseRandomProjection
from sklearn.feature_extraction.text import TfidfTransformer


class NeighborStrategy(object):
    __metaclass__ = abc.ABCMeta

    def _item_strategy(self, target_user, distances, indices,
                       zero_mean=False):
        indices = oneD(indices)
        distances = oneD(distances)
        if np.isscalar(target_user):
            ratings = np.array([self.database.get_rating(target_user, item, zero_mean)
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
                       zero_mean=False):
        indices = oneD(indices)
        distances = oneD(distances)
        ratings = np.array([self.database.get_rating(user, target_item, zero_mean)
                            for user in indices], ndmin=1)

        if self.metric == 'cosine':
            similarities = oneD((1.0 - distances))
        elif self.metric == 'correlation':
            similarities = - distances

        return (ratings, similarities)


class PredictionStrategy(object):
    __metaclass__ = abc.ABCMeta

    def _predict(self, ratings, similarities, target=None, zero_mean=False):
        if (ratings == 0).all():
            #print('All user ratings on neighbor items are zero')
            pred_rating = 0
        else:
            ratings = oneD(ratings)
            similarities = oneD(similarities)
            denominator = np.abs(similarities[ratings != 0]).sum()
            if denominator != 0:
                pred_rating = np.dot(ratings, similarities)/denominator
                if zero_mean:
                    pred_rating += self.database.get_means(zero_mean)[target]
            else:
                pred_rating = 0
        return pred_rating


class DummyRecommender(RatingPredictor):
    def __init__(self):
        self.user_means = []
        self.item_means = []
        self.database = None

    def fit(self, database):
        self.database = database
        matrix, means = self.database.get_matrix(zero_mean='useritems',
                                                      sparse=True)
        self.user_means = means['users']
        self.item_means = means['useritems']


    def predict(self, target_user, target_item):
        return self.user_means[target_user] + self.item_means[target_item]


class ItemBased(RatingPredictor, NeighborStrategy, PredictionStrategy):
    def __init__(self, n_neighbors=10, algorithm='brute', model_size=0.8,
                 metric='cosine', offline_kNN=True, weighting='none',
                 **kNN_args):
        self.database = None
        self.n_neighbors = n_neighbors
        self.model_size = model_size
        self.weighting = weighting
        self.metric = 'cosine'
        self.offline_kNN = offline_kNN
        self.kNN = neighbors.kNN(n_neighbors=self.model_size,
                                 algorithm=algorithm, metric=metric, **kNN_args)

    def fit(self, database):
        self.model_size = int(np.ceil(self.model_size * database.n_items()))
        self.kNN.estimator.n_neighbors = self.model_size
        self.database = database
        if self.offline_kNN:
            matrix, user_means = self.database.get_matrix(zero_mean='users',
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
            matrix, user_means = self.database.get_matrix(zero_mean='users')
            item_vector = self.database.get_item_vector(target_item, zero_mean='users')
            self.kNN.fit(matrix[:, rated_items].T)
            distances, indices = \
                self.kNN.kneighbors(item_vector, n_neighbors=n_neighbors)
            indices = oneD([rated_items[i] for i in oneD(indices)])

        if len(oneD(indices)) == 0:
            return 0

        ratings, similarities = \
            self._item_strategy(target_user, distances, indices,
                                zero_mean='items')

        return self._predict(ratings, similarities,
                             target_item, zero_mean='items')


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
            if self.metric == 'correlation':
                sparse = False
            else:
                sparse = True
            matrix, means = database.get_matrix(zero_mean='users',
                                                     sparse=sparse)
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
            matrix, user_means = self.database.get_matrix(zero_mean='users')
            self.kNN.fit(matrix[user_indices, :])
            user_vector = self.database.get_user_vector(target_user, zero_mean='users')
            n_neighbors = min(self.n_neighbors, len(user_indices))
            distances, indices = \
                self.kNN.kneighbors(user_vector, n_neighbors)
            indices = np.array([user_indices[i] for i in oneD(indices)])

        if len(oneD(indices)) == 0:
            return 0

        ratings, similarities = \
            self._user_strategy(target_item, distances, indices,
                                zero_mean='users')

        return self._predict(ratings, similarities, target_user, zero_mean='users')


class MFrecommender(RatingPredictor):
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


class MFNNrecommender(MFrecommender):
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

class BMFrecommender(MFNNrecommender, NeighborStrategy, PredictionStrategy):
    __MF_type__ = BMF

    def __MF_args__(RS_args):
        args = ['min_coverage', 'bin_threshold']
        return dict([(arg, RS_args[arg]) for arg in args])

    def __init__(self, neighbor_type='user', offline_kNN=False,
                 bin_threshold=0, min_coverage=1.0, weighting='none',
                 n_neighbors=10, algorithm='brute', metric='cosine',
                 model_size=0.8, **kNN_args):
        self.database = None
        self.weighting = weighting
        self.metric = metric
        self.neighbor_type = neighbor_type
        self.bin_threshold = bin_threshold
        self.min_coverage = min_coverage
        self.P = None
        self.Q = None
        self.n_neighbors = n_neighbors
        self.model_size = model_size
        # create a random seed that will be the same for a certain BMF,
        # to allow fair comparisons among other params
        self.seed = int(str(min_coverage).replace('.','') \
            + str(bin_threshold)) + RANDOM_SEED
        np.random.RandomState(self.seed)
        self.kNN = \
            neighbors.kNN(n_neighbors=n_neighbors,
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
                mf.fit(self.database.get_matrix(threshold=self.bin_threshold, sparse=True))
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
        elif self.weighting == 'factors':
            I = self.database.get_matrix(threshold=self.bin_threshold, sparse=True)
            if self.neighbor_type == 'user':
                self.P = np.dot(I, self.Q) \
                    /np.tile(self.Q.sum(axis=0), (self.P.shape[0], 1))
            elif self.neighbor_type == 'item':
                self.Q = np.dot(I.T, self.P) \
                    /np.tile(self.P.sum(axis=0), (self.Q.shape[0], 1))

        if self.offline_kNN:
            if self.neighbor_type == 'user':
                self.model_size = int(np.ceil(self.model_size * database.n_users()))
                self.kNN.estimator.n_neighbors = self.model_size
                self.kNN.fit(self.P, keepgraph=True)
            elif self.neighbor_type == 'item':
                self.model_size = int(np.ceil(self.model_size * database.n_items()))
                self.kNN.estimator.n_neighbors = self.model_size
                self.kNN.fit(self.Q, keepgraph=True)
            else:
                raise ValueError('Invalid neighbor_type "%s" parameter passed \
                                 to constructor' % self.neighbor_type)
        return self



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

        elif self.weighting == 'factors':
            if self.neighbor_type == 'user':
                self.P = np.dot(self.P, self.Q.T) \
                    /np.tile(self.Q.sum(axis=0), (self.P.shape[0], 1))
            elif self.neighbor_type == 'item':
                self.Q = np.dot(self.Q, self.P.T) \
                    /np.tile(self.P.sum(axis=0), (self.Q.shape[0], 1))

        if self.dim_red != 'auto':
            n_components = int(np.ceil(self.dim_red*self.P.shape[1]))
        else:
            n_components = self.dim_red

        np.random.RandomState(self.seed)
        if self.RP == 'gaussian':
            self.RP = GaussianRandomProjection(n_components=n_components)
        elif self.RP == 'sparse':
            self.RP = SparseRandomProjection(n_components=n_components)
        else:
            raise ValueError('Unrecognized RP type')

        np.random.RandomState(self.seed)
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

class SVDrecommender(MFrecommender):
    __MF_type__ = SVD

    def __MF_args__(RS_args):
        args = ['dim', 'regularization']
        return dict([(arg, RS_args[arg]) for arg in args])

    def __init__(self, dim=50, regularization=0.1):
        self.database = None
        self.P = None
        self.Q = None
        self.model = None
        self.dim = dim
        self.regularization = regularization

    def load_mf(self, P, Q, model):
        MFrecommender.load_mf(self, P, Q)
        self.model = model

    def gen_mf(self, database, **MF_args):
        ratings = []
        n_users = database.n_users()
        n_items = database.n_items()
        for user in range(n_users):
            ratings += [(user, i, r) for i, r in
                        database.get_rating_list(user, zero_mean=False)]
        self.model = SVD(ratings, n_users, n_items,
                         self.dim, self.regularization, **MF_args)
        self.model.optimize()
        self.P = self.model.users
        self.Q = self.model.items
        return self.P, self.Q, self.model

    def fit(self, database):
        if self.P is None or self.Q is None:
            self.gen_mf(database)

    def predict(self, target_user, target_item):
        return np.dot(self.P[target_user, :], self.Q[target_item, :])


class SVDNNrecommender(SVDrecommender, MFNNrecommender):
    def __init__(self, n_neighbors=20, model_size=1, offline_kNN=True,
                 dim=10, regularization=0.1):
        SVDrecommender.__init__(self, dim, regularization)
        self.n_neighbors = n_neighbors
        self.model_size = model_size
        self.kNN = None
        self.offline_kNN = self.offline_kNN

    def fit(self, database):
        SVDrecommender.fit(self, database)
        if self.offline_kNN:
            if self.neighbor_type == 'user':
                self.model_size = int(np.ceil(self.model_size * database.n_users()))
                self.kNN.estimator.n_neighbors = self.model_size
                self.kNN.fit(self.P, keepgraph=True)
            elif self.neighbor_type == 'item':
                self.model_size = int(np.ceil(self.model_size * database.n_items()))
                self.kNN.estimator.n_neighbors = self.model_size
                self.kNN.fit(self.Q, keepgraph=True)
            else:
                raise ValueError('Invalid neighbor_type "%s" parameter passed \
                                 to constructor' % self.neighbor_type)
        return self
