# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:10:13 2015

@author: thalita

Experiment 0

Test system with TestDB
"""

from databases import TestDB
from evaluation import kFoldView, kFoldEvaluator,\
    HoldoutRatingsEvaluator, HoldoutRatingsView
import recommender as rec
from multiprocessing import Pool
import os

PARALLEL = False

result_folder = 'results/exp_0_results/'
if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

RS_types = [rec.ItemBased,
            rec.BMFrecommender,
            rec.BMFRPrecommender]

database = TestDB(50, 100, min_items=0.2)
kfold_view = kFoldView(database, result_folder, n_folds=2)
holdout_view = HoldoutRatingsView(database, result_folder,
                                  pct_hidden=0.2, threshold=4)

def run(i):
    global kfold_view, RS_types, result_folder
    print('Running '+ str(RS_types[i].__name__))
    evalu = kFoldEvaluator(kfold_view, RS_types[i], {}, result_folder,
                           pct_hidden=0.2, threshold=3, topk=10)
    print('Training...')
    evalu.train(force_train=True)
    print('Done!')
    print('Testing...')
    evalu.test()
    print('Done!')
    evalu = HoldoutRatingsEvaluator(holdout_view, RS_types[i], {},
                                    result_folder, threshold=3, topk=10)
    print('Training...')
    evalu.train(force_train=True)
    print('Done!')
    print('Testing...')
    evalu.test(parallel=True)
    print('Done!')


if PARALLEL:
    pool = Pool()
    pool.map(run, [i for i in range(len(RS_types))])
else:
    for i in range(len(RS_types)):
        run(i)

