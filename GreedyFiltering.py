# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 17:10:52 2015

@author: thalita


"""

import numpy as np
import sklearn
from sklearn.metrics import pairwise_distances
import scipy
import scipy.sparse as sparse

import pprint as pp

def greedy_filtering(X, min_dim, metric='euclidean', method='brute'):
    """
    Greedy filtering
    Parameters:
    X : (n_samples, n_dim) array
    Output: k-NN queues for each vector

    """
    n_samples, n_dim = X.shape
    # formating vector list from X
    # vectors in vector_list must be in form:
    # [(value_i, dim(value_i)), ...]

    vector_list = []
    for i in range(n_samples):
        vector_list.append([])
        for j in range(n_dim):
            if X[i, j] > 0:
                vector_list[i].append((X[i, j], j))

    # prefix selection
    remaining = list(vector_list)

    dim_index = {}
    for d in range(n_dim):
        dim_index[d] = []

    pos = 0
    prefix = {}
    while remaining != []:
        for vec_id, vec in enumerate(remaining):
            value, dim = vec[pos]
            dim_index[dim].append(vec_id)

        for vec_id, vec in enumerate(remaining):
            dim_count = 0
            for j in range(pos):
                value, dim = vec[j]
                dim_count += len(dim_index[dim])
            if dim_count >= min_dim or pos >= len(vec)-1:
                prefix[vec_id] = pos
                remaining.pop(vec_id)
        pos += 1
    # Search
    if method == 'brute':
        knn = {}
        for vec_id, vec in enumerate(X):
            knn[vec_id] = set()

        for vec_ids in dim_index.values():
            print(vec_ids)
            D2v = dict([(i, v) for i, v in enumerate(vec_ids)])
            D = pairwise_distances(X[vec_ids,:], metric=metric)
            for i in range(D.shape[0]-1):
                for j in range(i+1, D.shape[1]):
                    print(i, j, D2v[i], D2v[j])
                    knn[D2v[i]].add((D[i, j], D2v[j]))
                    knn[D2v[j]].add((D[i, j], D2v[i]))
        for key in knn:
            knn[key] = sorted(list(knn[key]))

    elif method == 'inverted_index':
        raise ValueError(method+' not implemented yet.' )

    else:
        raise ValueError("Unknown search method")

    return knn
#%%
def __test__():
    matrix = np.array([[1, 0, 1, 0, 1, 1],
                       [0, 0, 1, 0, 0, 0],
                       [1, 1, 0, 1, 1, 1],
                       [0, 0, 1, 0, 0, 1],
                       [0, 1, 1, 1, 0, 1]])
    queues = greedy_filtering(matrix, min_dim=6)
    pp.pprint(queues)
__test__()
