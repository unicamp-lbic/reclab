# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:18:26 2015

@author: thalita
"""

from sklearn.neighbors import NearestNeighbors, LSHForest


class kNN(object):
    def __init__(self, n_neighbors=5,
                 algorithm='brute', metric='minkowski'):
        self.estimator = None
        if not algorithm == 'LSH':
            self.estimator = \
                NearestNeighbors(n_neighbors=n_neighbors, radius=1.0,
                                 algorithm=algorithm, leaf_size=30,
                                 metric=metric, p=2, metric_params=None)
        elif algorithm == 'LSH':
            self.estimator = \
                LSHForest(n_estimators=10, radius=1.0,
                          n_candidates=50, n_neighbors=n_neighbors,
                          min_hash_match=4, radius_cutoff_ratio=0.9,
                          random_state=None)

    def fit(self, X):
        return self.estimator.fit(X)

    def kneighbors(self, X, n_neighbors=None, return_distance=True):
        return self.estimator.kneighbors(X, n_neighbors, return_distance)
