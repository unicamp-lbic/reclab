# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:18:12 2015

@author: thalita
"""
from base import BaseDatabase
import numpy as np
import scipy.sparse as sparse
from utils import _get_zero_mean_matrix


class MatrixDatabase(BaseDatabase):
    def __init__(self, matrix):
        self.matrix = matrix

        self.thresholded = None

        self.zero_mean_matrix = None
        self.user_means = None

    def _compute_zero_mean(self):
        self.zero_mean_matrix, self.user_means = \
                    _get_zero_mean_matrix(self.matrix.copy())
    def n_users(self):
        return self.matrix.shape[0]

    def n_items(self):
        return self.matrix.shape[1]

    def get_matrix(self, zero_mean=False, threshold=None):
        if threshold is not None:
            if self.thresholded is None:
                self.thresholded = \
                    np.array(self.matrix > threshold, dtype=float)
            return self.thresholded
        elif zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            return self.zero_mean_matrix, self.user_means

        else:
            return self.matrix

    def get_rating(self, user_id, item_id, zero_mean=False):
        if zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            return self.zero_mean_matrix[user_id, item_id]
        else:
            return BaseDatabase.get_rating(self, user_id, item_id)

    def get_user_vector(self, user_id, zero_mean=False):
        if zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            return self.zero_mean_matrix[user_id, :]
        else:
            return BaseDatabase.get_user_vector(self, user_id)

    def get_rating_list(self, user_id):
        vector = BaseDatabase.get_user_vector(self, user_id)
        alist = [(r, i) for i, r in enumerate(vector) if r != 0]
        alist.sort()
        return [(i, r) for r, i in alist]

    def get_item_vector(self, item_id, zero_mean=False):
        if zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            return self.zero_mean_matrix[:, item_id]
        else:
            return BaseDatabase.get_item_vector(self, item_id)

    def get_unrated_items(self, user_id):
        "return unrated item ids for user"
        return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                if rating == 0]

    def get_rated_items(self, user_id):
        "return items rated by user_id user"
        return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                if rating > 0]

    def get_rated_users(self, item_id):
        "return users who did not rate item_id user"
        return [idx for idx, rating in enumerate(self.matrix[:, item_id])
                if rating > 0]


class HiddenRatingsDatabase(MatrixDatabase):
    def __init__(self, matrix, hidden_coord):
        MatrixDatabase.__init__(self, matrix.copy())
        for u, i in hidden_coord:
            self.matrix[u, i] = 0


class SubDatabase(MatrixDatabase):
    def __init__(self, matrix, user_indices):
        user_indices.sort()
        MatrixDatabase.__init__(self, matrix[user_indices, :])


class SubDatabaseOnline(MatrixDatabase):
    def __init__(self, matrix, user_indices):
        user_indices.sort()
        self.user_indices = user_indices

        MatrixDatabase.__init__(self, matrix)

    def _compute_zero_mean(self):
        self.zero_mean_matrix, self.user_means = \
            _get_zero_mean_matrix(self.matrix[self.user_indices, :].copy())

    def get_matrix(self, zero_mean=False, threshold=False):
        if threshold:
            if self.thresholded is None:
                self.thresholded = \
                    np.array(self.matrix[self.user_indices, :] > threshold,
                             dtype=float)
            return self.thresholded
        if zero_mean:
            self._compute_zero_mean()
            return self.zero_mean_matrix, self.user_means
        else:
            return self.matrix[self.user_indices, :]

    def n_users(self):
        return len(self.user_indices)

    def get_ratings(self, user_id, item_id, zero_mean=False):
        if zero_mean:
            self._compute_zero_mean()
            return self.zero_mean_matrix[user_id, item_id]
        else:
            return self.matrix[self.user_indices[user_id], item_id]

    def get_user_vector(self, user_id, zero_mean=False):
        "return a 2D array with user ratings"
        if zero_mean:
            self._compute_zero_mean()
            return self.zero_mean_matrix[user_id, :]
        else:
            return self.matrix[self.user_indices[user_id], :]

    def get_item_vector(self, item_id, zero_mean):
        "return a 2D array with item ratings"
        if zero_mean:
            self._compute_zero_mean()
            return self.zero_mean_matrix[:, item_id]
        else:
            return self.matrix[self.user_indices, item_id]

    def get_unrated_items(self, user_id):
        "return unrated item ids for user"
        return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                if rating == 0]

    def get_rated_items(self, user_id):
        "return items rated by user_id user"
        return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                if rating > 0]

    def get_rated_users(self, item_id):
        "return users who did rate item_id item"
        return [idx for idx, rating
                in enumerate(self.matrix[self.user_indices, item_id])
                if rating > 0]
