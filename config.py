# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:54:28 2015

@author: thalita

Config file
"""
from collections import defaultdict
import recommender as rec
import pandas as pd
import data

class Config(object):
    def __init__(self, database, RS_type, RS_args={},
                 nfolds=1, per_user=True, pct_hidden=None, threshold=None,
                 is_MF=False):
        self.database = data.base.STD_DB_NAMES[database]
        self.RS_type = RS_type
        self.RS_args = RS_args
        self.nfolds = nfolds
        self.per_user = per_user
        if nfolds == 1 and pct_hidden is None:
            raise ValueError('Holdout needs pct_hidden arg in config')
        self.pct_hidden = pct_hidden
        self.threshold = threshold
        self.is_MF = is_MF
        if is_MF:
            self.MF_type = RS_type.__MF_type__
            self.MF_args = RS_type.__MF_args__(RS_args)

    def as_dict(self):
        d = self.__dict__.copy()
        keys = list(d.keys())
        for key in keys:
            if d[key] is None:
                del d[key]

        del d['RS_args']
        d['RS_type'] = d['RS_type'].__name__
        d.update(self.RS_args)
        return d

dummy5fold = Config(
    database='ml100k',
    RS_type=rec.DummyRecommender,
    nfolds=5,
    threshold=3
)

IB5fold = Config(
    database='ml100k',
    RS_type=rec.ItemBased,
    RS_args={'n_neighbors': 20,
              'algorithm': 'brute',
              'metric': 'cosine'},
    nfolds=5,
    is_MF=False,
    threshold=3

)

BMF5fold = Config(
    database='ml100k',
    RS_type=rec.BMFrecommender,
    RS_args={'n_neighbors': 20,
              'neighborhood_type': 'user',
              'offline_kNN': False,
              'algorithm': 'brute',
              'metric': 'cosine',
              'min_coverage': 1.0,
              'bin_threshold': 3},
    nfolds=5,
    is_MF=True,
    threshold=3

)



'''
Dictionary of valid configuration settings
'''
valid_configs = {
    'BMF5fold': BMF5fold,
    'IB5fold': IB5fold,
    'dummy5fold': dummy5fold
}