# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 15:21:43 2015

@author: thalita


Experiment 4.1

Ensemble of BMF through RPs of different dimensions
re-using BMFRP from exp 3

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
from pickle import load

RP_folder = 'results/exp_3_results/'
trained_models = [RP_folder + f for f in os.listdir(RP_folder)
                  if f.find('.out') > -1]

result_folder = 'results/exp_4.1_results/'
database = MatrixDatabase(dbread.read_matrix())
holdout_view = HoldoutRatingsView(database, dbread.PATH, nsplits=1,
                                  pct_hidden=0.2, threshold=3)

if not os.path.isdir(result_folder):
    os.makedirs(result_folder)

RS_type = [ens.AvgRatingEnsemble, ens.RankSumEnsemble, ens.MajorityEnsemble]
dim_red = [0.25, 0.5, 0.75, 0.8, 0.9]
nneighbor = chain([5], range(10,61,10))
RP_type = ['sparse', 'gaussian']

args = [(nn, rp) for nn in nneighbor for rp in RP_type]
for nn, rp in args:
    files = [f for f in trained_models
             if f.find('nneighbor_%d'%nn) > -1
             if f.find('RPtype_%s'%rp) > -1]
    RS_list = []
    for d in dim_red:
        try:
            fname = [f for f in files
                     if f.find('dimred_%d'%d) > -1][0]
            with open(fname, 'rb') as f:
                RS_list.append(load(fname))
        except IndexError:
            pass

    def factory(**RS_args):
        global RS_list
        return RS_list

    RS_arguments = {'RS_factory': factory,
                    'n_neighbors': nn,
                    'threshold': 3,
                    'RP_type': rp}

    def run(i):
        print('Running %d' % i + str(RS_type[i]))
        evalu = HoldoutBMF(holdout_view, RS_type[i], RS_arguments,
                           result_folder, threshold=3, topk=20)

        print('Training %d' % i + str(RS_type[i]))
        evalu.train()
        print('Done training %d' % i + str(RS_type[i]))

        print('Testing %d' % i + str(RS_type[i]))
        evalu.test()
        print('Done testing %d' % i + str(RS_type[i]))
        try:
            pass
        except:
            with open(evalu.fname_prefix+'_error_log_%d.out' % i, 'w') as f:
                traceback.print_exception(*sys.exc_info(), file=f)

    for i in range(len(RS_type)):
        run(i)
