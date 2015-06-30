# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:04:53 2015

@author: thalita
"""

import data.MovieLens100k.dbread as dbread
from databases import MatrixDatabase
from evaluation import HoldoutBMF, HoldoutRatingsView
import recommender as rec
from itertools import chain
import os
import sys
import traceback


result_folder = 'results/exp_2_results/'
RS_type = rec.BMFrecommender

coverages = [1, 0.8, 0.6]
bin_thresh = [i for i in range(0, 5)]

RS_arguments = [{'neighbor_type': nn_type,
                 'offline_kNN': offline,
                 'n_neighbors': nn,
                 'algorithm': 'LSH',
                 'metric': 'cosine',
                 'threshold': t,
                 'min_coverage': coverage}
                for nn_type in ['user', 'item']
                for nn in chain([5], range(10, 61, 10))
                for t in bin_thresh
                for offline in [True, False]
                for coverage in coverages]


database = MatrixDatabase(dbread.read_matrix())
holdout_view = HoldoutRatingsView(database, dbread.PATH, nsplits=1,
                                  pct_hidden=0.2, threshold=3)

if not os.path.isdir(result_folder):
    os.makedirs(result_folder)


def run(i):
    global holdout_view, RS_type, RS_arguments, result_folder
    print('Running %d' % i + str(RS_arguments[i]))
    evalu = HoldoutBMF(holdout_view, RS_type, RS_arguments[i],
                       result_folder, threshold=3, topk=20)
    try:
        print('Training %d' % i + str(RS_arguments[i]))
        evalu.train()
        print('Done training %d' % i + str(RS_arguments[i]))

        print('Testing %d' % i + str(RS_arguments[i]))
        evalu.test()
        print('Done testing %d' % i + str(RS_arguments[i]))

    except:
        with open(evalu.fname_prefix+'_error_log_%d.out' % i, 'w') as f:
            traceback.print_exception(*sys.exc_info(), file=f)


for i in range(len(RS_arguments)):
    run(i)
