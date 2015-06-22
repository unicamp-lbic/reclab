# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 19:54:55 2015

@author: thalita

Generate BMFs with diferent coverages and binarization thresholds
"""


import data.MovieLens100k.dbread as dbread
from databases import MatrixDatabase
from evaluation import HoldoutBMF, HoldoutRatingsView
import recommender as rec
from multiprocessing import Pool
import sys
import traceback
from numpy.random import shuffle

if 'parallel' in set(sys.argv):
    PARALLEL = True
else:
    PARALLEL = False

coverages = [1, 0.8, 0.6]
bin_thresh = [i for i in range(0, 5)]

RS_type = rec.BMFrecommender
RS_arguments = [{'threshold': t,
                 'min_coverage': coverage}
                for t in bin_thresh
                for coverage in coverages]

shuffle(RS_arguments)

result_folder = '/dev/null'
database = MatrixDatabase(dbread.read_matrix())
holdout_view = HoldoutRatingsView(database, dbread.PATH, nsplits=1,
                                  pct_hidden=0.2, threshold=3)


def run(i):
    global holdout_view, RS_type, RS_arguments, result_folder
    print('Running %d' % i + str(RS_arguments[i]))
    evalu = HoldoutBMF(holdout_view, RS_type, RS_arguments[i],
                       result_folder, threshold=3, topk=20)
    try:
        print('Training %d' % i + str(RS_arguments[i]))
        evalu.train()
        print('Done training %d' % i + str(RS_arguments[i]))

    except:
        with open(evalu.fname_prefix+'_error_log_%d.out' % i, 'w') as f:
            traceback.print_exception(*sys.exc_info(), file=f)

if PARALLEL:
    pool = Pool()
    pool.map(run, range(len(RS_arguments)))
else:
    for i in range(len(RS_arguments)):
        run(i)
