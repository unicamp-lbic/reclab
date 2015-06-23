# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:18:26 2015

@author: thalita
"""

from sklearn.neighbors import NearestNeighbors, LSHForest
import numpy as np


class kNN(object):
    def __init__(self, n_neighbors=5,
                 algorithm='brute', metric='minkowski'):
        self.estimator = None
        self.graph = None
        if not algorithm == 'LSH':
            self.estimator = \
                NearestNeighbors(n_neighbors=n_neighbors, radius=1.0,
                                 algorithm=algorithm, leaf_size=30,
                                 metric=metric, p=2, metric_params=None)
        elif algorithm == 'LSH':
            if metric != 'cosine':
                raise ValueError('LSH forest can only use cosine metric')
            self.estimator = \
                LSHForest(n_estimators=10, radius=1.0,
                          n_candidates=50, n_neighbors=n_neighbors,
                          min_hash_match=4, radius_cutoff_ratio=0.9,
                          random_state=None)

    def fit(self, X, keepgraph=False):
        self.estimator = self.estimator.fit(X)
        if keepgraph:
            mat = \
                self.estimator.kneighbors_graph(
                    X, n_neighbors=min(X.shape[0], self.estimator.n_neighbors),
                    mode='distance')
            self.graph = {}
            for i, line in enumerate(mat):
                NN = [tup for tup in zip(line.data, line.indices)]
                NN.sort()
                self.graph[i] = [(ind, dist) for dist, ind in NN]
        return self.estimator

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        if np.isscalar(X) and self.graph is not None:
            if n_neighbors is None:
                select = self.graph[X][:self.estimator.n_neighbors]
            else:
                select = self.graph[X][:n_neighbors]
            indices = [i for i, d in select]
            if return_distance:
                distances = [d for i, d in select]
                return (distances, indices)
            else:
                return indices
        elif np.isscalar(X):
            raise ValueError('Must call fit with keepgraph=True to use graph')
        else:
            return self.estimator.kneighbors(X, n_neighbors, return_distance)

    def kneighbors_graph(self, X=None, n_neighbors=None, mode='distance'):
        return self.estimator.kneighbors_graph(X, n_neighbors, mode)
