# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:32:12 2015

@author: thalita

Experiment 1

Reproduce BMF results from Nenova et al (2013)
"""

import data.MovieLens100k.dbread as dbread
from databases import MatrixDatabase
from evaluation import HoldoutBMF, HoldoutRatingsView
import recommender as rec
from multiprocessing import Pool, Lock
from itertools import chain
import os
import sys
import traceback
from numpy.random import shuffle


if 'parallel' in set(sys.argv):
    PARALLEL = True
else:
    PARALLEL = False

result_folder = 'results/exp_1_results/'
RS_type = rec.BMFrecommender

coverages = [1, 0.8, 0.6]
bin_thresh = [i for i in range(0, 5)]

RS_arguments = [{'neighbor_type': nn_type,
                 'offline_kNN': offline,
                 'n_neighbors': nn,
                 'algorithm': 'brute',
                 'metric': 'cosine',
                 'threshold': t,
                 'min_coverage': coverage}
                for nn_type in ['user']
                for nn in chain([5], range(10, 61, 10))
                for t in bin_thresh
                for offline in [False]
                for coverage in coverages]
shuffle(RS_arguments)

BMF_locks = dict([(tup, Lock()) for tup in
                  [(c, t) for c in coverages for t in bin_thresh]])

database = MatrixDatabase(dbread.read_matrix())
holdout_view = HoldoutRatingsView(database, dbread.PATH, nsplits=1,
                                  pct_hidden=0.2, threshold=3)

if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

def run(i):
    global kfold_view, RS_type, RS_arguments, result_folder
    lock_key = (RS_arguments[i]['min_coverage'], RS_arguments[i]['threshold'])
    print('Running %d' % i + str(RS_arguments[i]))
    evalu = HoldoutBMF(holdout_view, RS_type, RS_arguments[i],
                       result_folder, threshold=3, topk=20)
    try:
        BMF_locks[lock_key].acquire()
        print('Training %d' % i + str(RS_arguments[i]))
        evalu.train()
        print('Done training %d' % i + str(RS_arguments[i]))
        BMF_locks[lock_key].release()

        print('Testing %d' % i + str(RS_arguments[i]))
        evalu.test()
        print('Done testing %d' % i + str(RS_arguments[i]))


    except:
        with open(evalu.fname_prefix+'_error_log_%d.out' % i, 'w') as f:
            traceback.print_exception(*sys.exc_info(), file=f)

if PARALLEL:
    pool = Pool()
    pool.map(run, range(len(RS_arguments)))
else:
    for i in range(len(RS_arguments)):
        run(i)
