# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import random_integers
import scipy.sparse as sparse


DB_PATHS = {
    'ml100k': 'data/MovieLens100k/',
    'TestDB': 'data/TestDB/'
}

STD_DB_NAMES= {
    'MovieLens100k':'ml100k',
    'ml100k': 'ml100k',
    'TestDB': 'TestDB',
    'testDB': 'TestDB'
}

'''
Specific db reading functions

Must be registered in the DB_READ dictionary
'''

def read_ml100k_matrix():
    PATH = DB_PATHS['ml100k']
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

def gen_testDB():
    return TestDB(100, 50, min_items=0.2)

'''
Dictionary of available DB read functions
'''
DB_READ = {
    'ml100k': read_ml100k_matrix,
    'TestDB': gen_testDB
}

def dbread(dbname):
    try:
        dbname = STD_DB_NAMES[dbname]
    except KeyError:
        raise KeyError('Database %s does not exist' % dbname)

    read_function = DB_READ[dbname]

    return read_function()

def get_db_path(dbname):
    try:
        dbname = STD_DB_NAMES[dbname]
    except KeyError:
        raise KeyError('Database %s does not exist' % dbname)
    return DB_PATHS[dbname]

def TestDB(nusers, nitems, min_items=0.02, binary=False):
    min_items = np.ceil(min_items*nitems)
    matrix = np.zeros((nusers, nitems))

    for i in range(nusers):
        extra = random_integers(0, min_items)
        total_items = max(min_items, min_items + extra)
        idx = random_integers(0, nitems-1, total_items)
        matrix[i, idx] = 1.0

    if not binary:
        matrix = np.multiply(matrix,
                             random_integers(1, 5, size=(nusers, nitems)))
        for i in range(0, nusers, 2):
            matrix[i+1, :] = np.ceil((matrix[i, :] + matrix[i+1, :]) / 2.0)
    return matrix