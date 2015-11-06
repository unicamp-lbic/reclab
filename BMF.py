# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:34:16 2015

@author: thalita
"""
import pyximport; pyximport.install()
import numpy as np
import scipy.sparse as sp
import time
from bmf_function import bmf, Concept


class BMF(object):
    def __init__(self, min_coverage=1.0, P=None, Q=None):
        self.P = P
        self.Q = Q
        self.min_coverage = min_coverage

    def fit(self, matrix):
        # if matrix has more cols than rows, factorize transposed matrix
        # (more efficient)
        nrow, ncol = matrix.shape
        if nrow < ncol:
            factors = bmf(matrix.T, self.min_coverage)
            self.Q, self.P = factors2matrices(factors, (ncol, nrow))
        else:
            factors = bmf(matrix, self.min_coverage)
            self.P, self.Q = factors2matrices(factors, (nrow, ncol))
        if sp.issparse(matrix):
            return self.P, self.Q
        else:
            return self.P.toarray(), self.Q.toarray()


def factors2matrices(factor_set, shape):
    n, m = shape
    d = len(factor_set)
    P = sp.lil_matrix((n, d))
    Q = sp.lil_matrix((m, d))
    for k, concept in enumerate(factor_set):
        for row in concept.extent:
            P[row, k] = 1
        for col in concept.intent:
            Q[col, k] = 1
    return P.tocsr(), Q.tocsr()

def bool_dot(m1, m2):
    m1 = m1.toarray().astype(bool)
    m2 = m2.toarray().astype(bool)
    result = np.dot(m1, m2)
    return result.astype(np.int)


def _test():
    matrix = np.array([[1, 0, 1, 0, 1, 1],
                       [0, 0, 1, 0, 0, 0],
                       [1, 1, 0, 1, 1, 1],
                       [0, 0, 1, 0, 0, 1],
                       [0, 1, 1, 1, 0, 1]])

    result = set([Concept([0, 2], [0, 4, 5]), Concept([2, 4], [1, 3, 5]),
                 Concept([0, 1, 3, 4], [2]), Concept([0, 2, 3, 4], [5])])

    t0 = time.time()
    factors = bmf(matrix)
    P, Q = factors2matrices(factors, matrix.shape)
    print('time:', time.time()-t0)
    print(factors)
    M = bool_dot(P, Q.T)
    assert((M == matrix).all())

    t0 = time.time()
    sparse_mat = sp.lil_matrix(matrix)
    factors = bmf(sparse_mat)
    P, Q = factors2matrices(factors, matrix.shape)
    print('time:', time.time()-t0)
    print(factors)
    M = bool_dot(P, Q.T)
    assert((M == matrix).all())


    print(P)
    print(Q.T)

def _large_test():
    shape = (1000, 50)
    matrix = np.random.binomial(n=1, p=0.2, size=shape )
    t0 = time.time()
    factors = bmf(matrix)
    P, Q = factors2matrices(factors, matrix.shape)
    print('time:', time.time()-t0)
    M = bool_dot(P, Q.T)
    assert((M == matrix).all())
    print(P.shape)

    t0 = time.time()
    sparse_mat = sp.lil_matrix(matrix)
    factors = bmf(sparse_mat)
    P, Q = factors2matrices(factors, matrix.shape)
    print('time:', time.time()-t0)
    M = bool_dot(P, Q.T)
    assert((M == matrix).all())

_test()