# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 17:10:46 2015

@author: thalita
"""
import numpy as np
import scipy.sparse as sp


class Concept(object):
    def __init__(self, extent=[], intent=[]):
        self.extent = set(extent)
        self.intent = set(intent)

    def __repr__(self):
        return self.extent.__repr__() + self.intent.__repr__()

    def copy(self):
        return Concept(self.extent.copy(), self.intent.copy())


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