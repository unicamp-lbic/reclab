# -*- coding: utf-8 -*-

import numpy as np
import scipy.sparse as sparse


DB_PATHS = {
    'ml100k': 'data/MovieLens100k/'
}

STD_DB_NAMES= {
    'MovieLens100k':'ml100k',
    'ml100k': 'ml100k',
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

'''
Dictionary of available DB read functions
'''
DB_READ = {
    'ml100k': read_ml100k_matrix
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