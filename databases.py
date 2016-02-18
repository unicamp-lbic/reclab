# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 12:18:12 2015

@author: thalita
"""
from base import BaseDatabase
import numpy as np
import scipy.sparse as sp
from utils import oneD


def _get_zero_mean_matrix(matrix, along='users'):
    rows, cols = matrix.shape
    if along == 'users':
        if sp.issparse(matrix):
            matrix = sp.csr_matrix(matrix, dtype=np.float)
        else:
            matrix = np.array(matrix, dtype=np.float)
        mean_vals = np.zeros(rows)
        for i in range(rows):
            if sp.issparse(matrix):
                mean = np.mean(matrix[i,:].data)
                mean_vals[i] = mean if not np.isnan(mean) else 0
                matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] \
                    -= mean_vals[i]
            else:
                non_zero_pos = matrix[i, :] > 0
                non_zero = matrix[i, non_zero_pos]
                mean = np.mean(non_zero)
                mean_vals[i] = mean if not np.isnan(mean) else 0
                matrix[i, non_zero_pos] -= mean_vals[i]
    else:  # Along Items
        if sp.issparse(matrix):
            matrix = sp.csc_matrix(matrix, dtype=np.float)
        else:
            matrix = np.array(matrix, dtype=np.float)
        mean_vals = np.zeros(cols)
        for i in range(cols):
            if sp.issparse(matrix):
                mean = np.mean(matrix[:, i].data)
                mean_vals[i] = mean if not np.isnan(mean) else 0
                matrix.data[matrix.indptr[i]:matrix.indptr[i+1]] \
                    -= mean_vals[i]
            else:
                non_zero_pos = matrix[:, i] > 0
                non_zero = matrix[non_zero_pos, i]
                mean = np.mean(non_zero)
                mean_vals[i] = mean if not np.isnan(mean) else 0
                matrix[non_zero_pos, i] -= mean_vals[i]
    if sp.issparse(matrix):
        matrix = matrix.tocsr()
    return matrix, mean_vals

class MatrixDatabase(BaseDatabase):
    def __init__(self, matrix):
        if  sp.issparse(matrix):
            matrix = matrix.tocsr()

        self.matrix = matrix
        self.matrix_csc = None
        self.matrix_dok = None

        self.thresholded = None
        self.thresholded_csc = None

        self.zero_mean_matrix = None
        self.zero_mean_matrix_csc = None
        self.zero_mean_matrix_dok = None

        self.means = {}

    def _compute_zero_mean(self):
        self.zero_mean_matrix = {}
        self.zero_mean_matrix['users'], self.means['users'] = \
            _get_zero_mean_matrix(self.matrix.copy(), along='users')
        self.zero_mean_matrix['items'], self.means['items'] = \
            _get_zero_mean_matrix(self.matrix.copy(), along='items')
        self.zero_mean_matrix['useritems'], self.means['useritems'] = \
            _get_zero_mean_matrix(self.zero_mean_matrix['users'].copy(), along='items')
        if sp.issparse(self.matrix):
            self.zero_mean_matrix_csc = {}
            self.zero_mean_matrix_dok= {}
            for name in ['users', 'items', 'useritems']:
                self.zero_mean_matrix_csc[name] = self.zero_mean_matrix[name].tocsc()
                self.zero_mean_matrix_dok[name] = self.zero_mean_matrix[name].todok()

    def get_means(self, along):
        if self.zero_mean_matrix is None:
            self._compute_zero_mean()
        return self.means[along]

    def n_users(self):
        return self.matrix.shape[0]

    def n_items(self):
        return self.matrix.shape[1]

    def get_matrix(self, zero_mean=False, threshold=None, sparse=False):
        if threshold is not None:
            if self.thresholded is None:
                if sp.issparse(self.matrix):
                    self.thresholded = self.matrix.copy()
                    self.thresholded.data = np.array(self.matrix.data > threshold,
                                                     dtype=np.float)
                    if sparse:
                        return self.thresholded
                    else:
                        return self.thresholded.toarray()
                else:
                    self.thresholded = \
                        np.array(self.matrix > threshold, dtype=np.float)
                    return self.thresholded
        elif zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            if sp.issparse(self.matrix) and not sparse:
                return self.zero_mean_matrix[zero_mean].toarray(), self.means
            else:
                return self.zero_mean_matrix[zero_mean], self.means
        else:
            if sp.issparse(self.matrix):
                if sparse:
                    return self.matrix
                else:
                    return self.matrix.toarray()
            else:
                return self.matrix

    def get_rating(self, user_id, item_id, zero_mean=False):
        if zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            if sp.issparse(self.matrix):
                return self.zero_mean_matrix_dok[zero_mean][user_id, item_id]
            else:
                return self.zero_mean_matrix[zero_mean][user_id, item_id]
        else:
            if sp.issparse(self.matrix):
                if self.matrix_dok is None:
                    self.matrix_dok = self.matrix.todok()
                return self.matrix_dok[user_id, item_id]
            else:
                return BaseDatabase.get_rating(self, user_id, item_id)

    def set_rating(self, user_id, item_id, rating):
        self.matrix[user_id, item_id] = rating
        self.matrix_csc = None
        self.matrix_dok = None

        self.thresholded = None
        self.thresholded_csc = None

        self.zero_mean_matrix = None
        self.zero_mean_matrix_csc = None
        self.zero_mean_matrix_dok = None

        self.means = {}

    def get_user_vector(self, user_id, zero_mean=False, sparse=False):
        if zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()

            vector = self.zero_mean_matrix[zero_mean][user_id, :]
        else:
            vector = BaseDatabase.get_user_vector(self, user_id)

        if sp.issparse(self.matrix) and not sparse:
            return oneD(vector.toarray())
        else:
            return vector

    def get_rating_list(self, user_id, zero_mean=False):
        if sp.issparse(self.matrix):
            if zero_mean:
                if self.zero_mean_matrix is None:
                    self._compute_zero_mean()
                ratings = self.zero_mean_matrix[user_id, :].data.tolist()
            else:
                ratings = self.matrix[user_id, :].data.tolist()
            items = self.matrix[user_id, :].indices.tolist()
            alist = list(zip(ratings, items))
        else:
            vector = self.get_user_vector(user_id, zero_mean)
            alist = [(r, i) for i, r in enumerate(vector) if r != 0]
        alist.sort()
        return [(i, r) for r, i in alist]

    def get_item_vector(self, item_id, zero_mean=False, sparse=False):
        if zero_mean:
            if self.zero_mean_matrix is None:
                self._compute_zero_mean()
            if sp.issparse(self.matrix):
                if sparse:
                    self.zero_mean_matrix_csc[zero_mean][:, item_id]
                else:
                    return oneD(self.zero_mean_matrix_csc[zero_mean][:, item_id].toarray())
            else:
                return self.zero_mean_matrix_csc[zero_mean][:, item_id]
        else:
            if sp.issparse(self.matrix):
                if self.matrix_csc is None:
                    self.matrix_csc = self.matrix.tocsc()
                if sparse:
                    return self.matrix_csc[:, item_id]
                else:
                    return oneD(self.matrix_csc[:, item_id].toarray())
            else:
                return BaseDatabase.get_item_vector(self, item_id)

    def get_unrated_items(self, user_id):
        "return unrated item ids for user"
        if sp.issparse(self.matrix):
            rated = set(self.matrix[user_id, :].indices)
            return [item for item in range(self.n_items())
                    if item not in rated]
        else:
            return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                    if rating == 0]

    def get_rated_items(self, user_id):
        "return items rated by user_id user"
        if sp.issparse(self.matrix):
            return self.matrix[user_id,:].indices.tolist()
        else:
            return [idx for idx, rating in enumerate(self.matrix[user_id, :])
                    if rating > 0]

    def get_rated_users(self, item_id):
        "return users who did not rate item_id user"
        if sp.issparse(self.matrix):
            if self.matrix_csc is None:
                self.matrix_csc = self.matrix.tocsc()
            return self.matrix_csc[:, item_id].indices.tolist()
        else:
            return [idx for idx, rating in enumerate(self.matrix[:, item_id])
                    if rating > 0]

def _test_sparse_matrixdb():
    sm = sp.rand(100, 200, density=0.02)
    m = sm.toarray()
    sdb = MatrixDatabase(sm)
    db = MatrixDatabase(m)
    assert((db.get_matrix() == sdb.get_matrix()).all())
    assert((db.get_matrix(zero_mean='users')[0] == sdb.get_matrix(zero_mean='users')[0]).all())
    assert((db.get_matrix(zero_mean='items')[0] == sdb.get_matrix(zero_mean='items')[0]).all())
    assert((db.get_matrix(zero_mean='users')[1] == sdb.get_matrix(zero_mean='users')[1]).all())
    assert((db.get_matrix(zero_mean='items')[1] == sdb.get_matrix(zero_mean='items')[1]).all())
    assert((db.get_matrix(threshold=0)==sdb.get_matrix(threshold=0)).all())
    assert((db.get_rating(5,30)==sdb.get_rating(5,30)).all())
    assert((db.get_rating(5,30,zero_mean='users')==sdb.get_rating(5,30,zero_mean='users')).all())
    assert((db.get_rating(5,30,zero_mean='items')==sdb.get_rating(5,30,zero_mean='items')).all())
    assert((db.get_user_vector(5)==sdb.get_user_vector(5)).all())
    assert((db.get_user_vector(5,zero_mean='users')==sdb.get_user_vector(5,zero_mean='users')).all())
    assert((db.get_user_vector(5,zero_mean='items')==sdb.get_user_vector(5,zero_mean='items')).all())
    assert((db.get_item_vector(5)==sdb.get_item_vector(5)).all())
    assert(db.get_unrated_items(5)==sdb.get_unrated_items(5))
    assert(db.get_rated_items(5)==sdb.get_rated_items(5))
    assert(db.get_rated_users(5)==sdb.get_rated_users(5))


class HiddenRatingsDatabase(MatrixDatabase):

    def __init__(self, matrix, hidden_coord):
        MatrixDatabase.__init__(self, matrix.copy())
        if sp.issparse(self.matrix):
            self.matrix = self.matrix.tolil()
        for u, i in hidden_coord:
            self.matrix[u, i] = 0
        if sp.issparse(self.matrix):
            self.matrix = self.matrix.tocsr()


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
