# -*- coding: utf-8 -*-
"""
Created on Mon Jun  1 15:11:28 2015

@author: thalita

Experiment 3

BMF + RP and regular kNN
"""


import data.MovieLens100k.dbread as dbread
from databases import MatrixDatabase
from evaluation import HoldoutBMF, HoldoutRatingsView
import recommender as rec
from multiprocessing import Pool, Lock
from itertools import chain
import os
import sys


if 'parallel' in set(sys.argv):
    PARALLEL = True
else:
    PARALLEL = False

result_folder = 'results/exp_3_results/'
RS_type = rec.BMFRPrecommender

coverages = [1, 0.8, 0.6]

RS_arguments = [{'neighbor_type': nn_type,
                 'offline_kNN': offline,
                 'n_neighbors': nn,
                 'algorithm': 'brute',
                 'metric': 'cosine',
                 'threshold': t,
                 'min_coverage': coverage}
                for nn_type in ['user', 'item']
                for nn in chain([5], range(10, 61, 10))
                for t in range(0, 4)
                for offline in [True, False]
                for coverage in coverages]

BMF_locks = dict([(i, Lock()) for i in coverages])

database = MatrixDatabase(dbread.read_matrix())
holdout_view = HoldoutRatingsView(database, dbread.PATH,
                                  pct_hidden=0.2, threshold=4)

if not os.path.isdir(result_folder):
    os.makedirs(result_folder)


def run(i):
    global kfold_view, RS_type, RS_arguments, result_folder
    min_coverage = RS_arguments[i]['min_coverage']
    print('Running %d' % i + str(RS_arguments[i]))
    evalu = HoldoutBMF(holdout_view, RS_type, RS_arguments[i],
                       result_folder, threshold=3, topk=10)

    BMF_locks[min_coverage].acquire()
    print('Training %d' % i + str(RS_arguments[i]))
    evalu.train()
    print('Done training %d' % i + str(RS_arguments[i]))
    BMF_locks[min_coverage].release()

    print('Testing %d' % i + str(RS_arguments[i]))
    evalu.test()
    print('Done testing %d' % i + str(RS_arguments[i]))


if PARALLEL:
    pool = Pool()
    pool.map(run, range(len(RS_arguments)))
else:
    for i in range(len(RS_arguments)):
        run(i)
