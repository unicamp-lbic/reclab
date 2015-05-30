# -*- coding: utf-8 -*-
"""
Created on Tue May 19 15:13:13 2015

@author: thalita
"""
import numpy as np
import scipy.sparse as sparse

PATH = 'data/MovieLens100k/'

def read_matrix():
    qtty = {}
    with open(PATH + 'u.info', 'r') as f:
        for line in f.readlines():
            value, label = tuple(line.split())
            qtty[label] = int(value)

    with open(PATH + 'u.data', 'r') as f:
        # base is in format: user_ID  item_ID  rating (tab separated)
        ratings = [tuple([int(elem) for elem in line.split('\t')[0:-1]])
                   for line in f]

    i, j, data = zip(*ratings)
    # remember user and item are 1-indexed in original file
    i = np.array(i)-1
    j = np.array(j)-1
    matrix = sparse.coo_matrix((data,(i, j)),
                               shape=(qtty['users'], qtty['items']))
    return matrix.toarray()