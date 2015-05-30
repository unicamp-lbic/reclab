# -*- coding: utf-8 -*-
"""
Created on Thu May 21 14:32:12 2015

@author: thalita

Experiment 1

Reproduce BMF results from Nenova et al (2013)
"""

import data.MovieLens100k.dbread as dbread
from databases import MatrixDatabase
from evaluation import HoldoutRatingsEvaluator, HoldoutRatingsView
import recommender as rec
from multiprocessing import Pool
from itertools import chain
import os


PARALLEL = False

result_folder = 'results/exp_1_results/'
RS_type = rec.BMFrecommender
RS_arguments = [{'neighbor_type': 'user',
                 'offline_kNN': offline,
                 'n_neighbors': nn,
                 'algorithm': 'brute',
                 'metric': 'cosine',
                 'threshold': t,
                 'min_coverage': coverage}
                for nn in chain([5], range(10, 61, 10))
                for t in range(0, 4)
                for offline in [True, False]
                for coverage in [1, 0.8, 0.6]]

database = MatrixDatabase(dbread.read_matrix())
holdout_view = HoldoutRatingsView(database, dbread.PATH,
                                  pct_hidden=0.2, threshold=4)

if not os.path.isdir(result_folder):
    os.makedirs(result_folder)
    
def run(i):
    global kfold_view, RS_type, RS_arguments, result_folder
    print('Running ' + str(RS_arguments[i]))
    evalu = HoldoutRatingsEvaluator(holdout_view, RS_type, RS_arguments[i],
                                    result_folder, threshold=3, topk=10)
    print('Training...')
    evalu.train()
    print('Done!')
    print('Testing...')
    evalu.test()
    print('Done!')


if PARALLEL:
    pool = Pool()
    pool.map(run, range(len(RS_arguments)))
else:
    for i in range(len(RS_arguments)):
        run(i)
