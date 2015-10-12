# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:34:16 2015

@author: thalita
"""
import numpy as np
import scipy.sparse as sp
import time


class Concept(object):
    def __init__(self, extent=[], intent=[]):
        self.extent = set(extent)
        self.intent = set(intent)

    def __repr__(self):
        return self.extent.__repr__() + self.intent.__repr__()

    def copy(self):
        return Concept(self.extent.copy(), self.intent.copy())


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

def bmf(matrix, min_coverage=1.0):
    n, m = matrix.shape
    if sp.issparse(matrix):
        remaining_pairs = sp.lil_matrix(matrix)
    else:
        remaining_pairs = matrix.copy()

    max_uncoverage = (1 - min_coverage) * remaining_pairs.sum()
    factors = set()
    while remaining_pairs.sum() > max_uncoverage :
        curr_concept = Concept()
        curr_coverage = 0
        # look greedily for maximal coverage concept
        while True:
            new_coverage = [0]*m  # indexed by attributes in 0 to m-1
            coverage = lambda concept: \
                remaining_pairs[list(concept.extent), :]\
                               [:, list(concept.intent)].sum()
            concepts = [curr_concept.copy() for i in range(m)]
            # Generate possible concepts
            for j in range(m):
                if j not in curr_concept.intent:
                    # add atribute j
                    concepts[j].intent.add(j)
                    # update/generate extent
                    if concepts[j].extent != set():
                        concepts[j].extent = \
                            set([obj for obj in concepts[j].extent
                             if matrix[obj, j] == 1])
                    else:
                        concepts[j].extent = \
                            set([obj for obj in range(n)
                             if matrix[obj, j] == 1])
                    # Update intent
                    update_intent = np.ones((1,m))
                    for obj in concepts[j].extent:
                        if sp.issparse(remaining_pairs):
                            update_intent *= remaining_pairs[obj, :].toarray()
                        else:
                            update_intent *= remaining_pairs[obj, :]
                        if update_intent.sum() == 0:
                            break
                    update_intent = [i for i, val in enumerate(update_intent[0,:])
                                    if val == 1]
                    concepts[j].intent.update(update_intent)
                    # calculate coverage
                    new_coverage[j] = coverage(concepts[j])

            # Pick concept with maximal coverage
            # Exit if coverage is not improved
            max_coverage = max(new_coverage)
            if max_coverage > curr_coverage:
                attribute = np.argmax(new_coverage)
                curr_concept = concepts[attribute].copy()
                curr_coverage = max_coverage
            else:
                break

        #  add curr_concept to factor set
        factors.add(curr_concept)
        for i in curr_concept.extent:
            for j in curr_concept.intent:
                remaining_pairs[i, j] = 0

    return factors


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
    m1 = m1 > 0
    m2 = m2 > 0
    return np.dot(m1, m2).astype(np.int)


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
