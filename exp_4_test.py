# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 10:58:08 2015

@author: thalita

Experiment 4

Ensemble of BMF through RPs of different dimensions

"""

import data.MovieLens100k.dbread as dbread
from databases import MatrixDatabase
from evaluation import HoldoutBMF, HoldoutRatingsView
import recommender as rec
import ensemble as ens
from itertools import chain
import os
import sys
import traceback


result_folder = 'results/exp_4_results/'

RS_type = ens.AvgRatingEnsemble

RS_arguments = [{'RS_factory': ens.RPBMFEnsembleFactory,
                 'RP_type': rp,
                 'n_projections': proj,
                 'dim_range': dimrange,
                 'neighbor_type': nn_type,
                 'n_neighbors': nn,
                 'threshold': t}
                for rp in ['sparse', 'gaussian']
                for nn_type in ['user']
                for nn in chain([5], range(10, 61, 10))
                for t in [3]
                for offline in [False]
                for coverage in [1]
                for dimrange in [(0.25, 0.75)]
                for proj in chain([3, 5], range(10, 31, 10))]

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

    print('Training %d' % i + str(RS_arguments[i]))
    evalu.train()
    print('Done training %d' % i + str(RS_arguments[i]))

    print('Testing %d' % i + str(RS_arguments[i]))
    evalu.test()
    print('Done testing %d' % i + str(RS_arguments[i]))
    try:
        pass
    except:
        with open(evalu.fname_prefix+'_error_log_%d.out' % i, 'w') as f:
            traceback.print_exception(*sys.exc_info(), file=f)


for i in range(len(RS_arguments)):
    run(i)
