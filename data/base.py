# -*- coding: utf-8 -*-

import numpy as np
from numpy.random import random_integers
import scipy.sparse as sp
import pandas as pd


DB_PATHS = {
    'ml100k': 'data/MovieLens100k/',
    'TestDB': 'data/TestDB/',
    'delicious': 'data/Delicious2k/'
}

STD_DB_NAMES= {
    'MovieLens100k':'ml100k',
    'ml100k': 'ml100k',
    'TestDB': 'TestDB',
    'testDB': 'TestDB',
    'delicious': 'delicious'
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
    matrix = sp.coo_matrix((data,(i, j)),
                               shape=(qtty['users'], qtty['items']))
    return matrix

def read_delicious():
    nusers =  1867
    nitems = 69226
    PATH = DB_PATHS['delicious']
    data = set()
    users = {}
    usercount = 0
    items = {}
    itemcount = 0
    with open(PATH +  'user_taggedbookmarks.dat', 'r') as f:
        for line in f:
            elems = line.split('\t')
            try:
                user = int(elems[0])
                if user not in users:
                    users[user] = usercount
                    usercount += 1
                bookmark = int(elems[1])
                if bookmark not in items:
                    items[bookmark] = itemcount
                    itemcount += 1
                data.add((users[user], items[bookmark]))
            except ValueError:
                pass

    pd.to_pickle({'usermap':users,'itemmap': items},
                 PATH + 'useritemmapping.pkl')
    data = list(data)
    row = np.array([u for u, i in data])
    col = np.array([i for u, i in data])
    data = np.array([1 for u, i in data])
    matrix = sp.coo_matrix((data, (row, col)),
                        shape=(nusers, nitems))
    return matrix.tocsr()

def gen_testDB():
    return TestDB(200, 100, min_items=0.2)

'''
Dictionary of available DB read functions
'''
DB_READ = {
    'ml100k': read_ml100k_matrix,
    'TestDB': gen_testDB,
    'delicious': read_delicious
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

def TestDB(nusers, nitems, min_items=0.02, binary=False, sparse=False):
    if sparse == True:
        delta = lambda : random_integers(0, np.ceil(min_items*nitems))/nitems
        mat = [sp.rand(1, nitems, density=min_items+delta(),format='csr')] * nusers
        matrix = sp.vstack(mat).tocsr()

    else:
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