# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:10:13 2015

@author: thalita

Experiment 0

Test system with TestDB
"""

from databases import TestDB
from evaluation import kFoldView, kFoldEvaluator
import recommender as rec
from multiprocessing import Pool

PARALLEL = False

result_folder = 'results/exp_0_results/'
RS_types = [rec.ItemBased, rec.BMFrecommender, rec.BMFRPrecommender]

database = TestDB(10, 40, min_items=0.5)
kfold_view = kFoldView(database, result_folder, n_folds=2)

def run(i):
    global kfold_view, RS_types, result_folder
    print('Running '+ str(RS_types[i].__name__))
    evalu = kFoldEvaluator(kfold_view, RS_types[i], {}, result_folder,
                           pct_hidden = 0.2, threshold = 3, topk=10)
    evalu.train(force_train=True)
    evalu.test()
    evalu = HoldoutRatingsEvaluator()

for i in range(len(RS_types)):
    run(i)

#if PARALLEL:
#    pool = Pool()
#    pool.map(run, [i for i in range(len(RS_types))])
#else:
#    map(run, [i for i in range(len(RS_types))])

